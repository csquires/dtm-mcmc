from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DARK_GREY = (.3, .3, .3)
LIGHT_GREY = (.5, .5, .5)

cmap = plt.get_cmap('hot')
Writer = writers['ffmpeg']
writer = Writer(fps=1)


def sites2rects(sites, m):
    rects = []
    for lon, lat in sites:
        x, y = m(lon, lat)
        x1, y1 = m(lon+1, lat+1)
        w = x1 - x
        h = y1 - y
        rects.append(Rectangle(xy=(x, y), width=w, height=h))
    return rects


def setup_map(sites, ax=None):
    min_lon = min(lon for lon, lat in sites)
    min_lat = min(lat for lon, lat in sites)
    max_lon = max(lon for lon, lat in sites)
    max_lat = max(lat for lon, lat in sites)

    m = Basemap(
        projection='merc',
        llcrnrlat=min_lat,
        llcrnrlon=min_lon,
        urcrnrlat=max_lat,
        urcrnrlon=max_lon,
        ax=ax
    )
    m.drawcoastlines()
    m.fillcontinents(color=LIGHT_GREY, lake_color=DARK_GREY)
    m.drawmapboundary(fill_color=DARK_GREY)
    return m


def create_topics_snapshot(etas, sites2ixs, topic, time, title=None, filename=None, fig=None, ax=None):
    if fig is None and ax is None:
        raise ValueError("fig or ax must not be None")
    if ax is None:
        ax = fig.gca()

    sites = sites2ixs.keys()
    m = setup_map(sites, ax=ax)
    rects = sites2rects(sites, m)

    if title is not None:
        ax.text(.01, .95, title, color='w', transform=ax.transAxes)

    # plot sites
    for site, rect in zip(sites, rects):
        ax.add_patch(rect)
        ix = sites2ixs[site][time]
        col = cmap(etas[time][ix, topic]) if ix is not None else cmap(.5)
        rect.set_color(col)

    # save
    if filename is not None:
        if fig is not None:
            fig.savefig(filename, bbox_inches='tight')
        else:
            raise ValueError("can't save if passed ax instead of fig")

    # return
    if fig is not None:
        return fig
    else:
        return ax


def create_topics_matrix(
        etas,
        sites2ixs,
        times=None,
        topics=None,
        time_names=None,
        filename=None,
        colors=None
):
    T = len(etas)
    K = etas[0].shape[1]
    times = times if times is not None else range(T)
    topics = topics if topics is not None else range(K)

    FONTSIZE = 5

    figsize = (len(times), len(topics))
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(nrows=len(topics), ncols=len(times))

    for i, k in enumerate(topics):
        for j, t in enumerate(times):
            ax = axes[i][j]
            if i == len(topics) - 1 and time_names is not None:
                ax.set_xlabel(time_names[t])
            if j == 0:
                ax.set_ylabel(k+1, rotation=0, ha='right', labelpad=6)
            create_topics_snapshot(etas, sites2ixs, k, t, ax=ax)
            if colors is not None:
                print(colors[t][k])
                for s in ['top', 'bottom', 'left', 'right']:
                    ax.spines[s].set_edgecolor(colors[t][k])
                    ax.spines[s].set_linewidth(2)

    plt.subplots_adjust(
        wspace=.05,
        hspace=.05,
        left=.08,
        right=.96,
        bottom=.10,
        top=.90
    )

    if filename is not None:
        plt.savefig(filename, dpi=300)

    return plt


def create_topics_animation(etas, sites2ixs, filename, topic, titles):
    T = len(etas)
    sites = sites2ixs.keys()
    m = setup_map(sites2ixs)

    fig = plt.figure()

    rects = sites2rects(sites, m)
    ax = fig.gca()
    for rect in rects:
        ax.add_patch(rect)
    title = plt.text(.01, .95, '', color='w', transform=ax.transAxes)

    def init():
        for rect in rects:
            rect.set_color('grey')
        return rects

    def animate(t):
        print(t)
        title.set(text=titles[t])
        for site, rect in zip(sites, rects):
            ix = sites2ixs[site][t]
            if ix is not None:
                e = etas[t][ix, topic]
                # print(e)
                col = cmap(e)
            else:
                col = cmap(.5)
            rect.set_color(col)
        return rects

    anim = FuncAnimation(fig, animate, init_func=init, frames=T, interval=2000, blit=True)
    anim.save(filename, writer)



