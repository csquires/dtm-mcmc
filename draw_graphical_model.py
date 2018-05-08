import daft
from matplotlib import rc
rc('font', family='serif', size=8)
rc('patch', facecolor='white', linewidth=0)
# rc('text', usetex=True)

x_unit = 2
x1 = 1*x_unit
x2 = 2*x_unit
x3 = 3*x_unit
pgm = daft.PGM([4*x_unit, 7], node_unit=1.2)

TOP = 6
pgm.add_node(daft.Node('alpha1', r'$\alpha_{t-1}$', x=x1, y=TOP))
pgm.add_node(daft.Node('alpha2', r'$\alpha_{t}$', x=x2, y=TOP))
pgm.add_node(daft.Node('alpha3', r'$\alpha_{t+1}$', x=x3, y=TOP))
pgm.add_edge('alpha1', 'alpha2')
pgm.add_edge('alpha2', 'alpha3')

pgm.add_node(daft.Node('eta1', r'$\eta_{t-1, d}$', x=x1, y=TOP-1))
pgm.add_node(daft.Node('eta2', r'$\eta_{t, d}$', x=x2, y=TOP-1))
pgm.add_node(daft.Node('eta3', r'$\eta_{t+1, d}$', x=x3, y=TOP-1))
pgm.add_edge('alpha1', 'eta1')
pgm.add_edge('alpha2', 'eta2')
pgm.add_edge('alpha3', 'eta3')

pgm.add_node(daft.Node('z1', r'$z_{t-1, d, n}$', x=x1, y=TOP-2))
pgm.add_node(daft.Node('z2', r'$z_{t, d, n}$', x=x2, y=TOP-2))
pgm.add_node(daft.Node('z3', r'$z_{t+1, d, n}$', x=x3, y=TOP-2))
pgm.add_edge('eta1', 'z1')
pgm.add_edge('eta2', 'z2')
pgm.add_edge('eta3', 'z3')

pgm.add_node(daft.Node('w1', r'$w_{t-1, d, n}$', x=x1, y=TOP-3, observed=True))
pgm.add_node(daft.Node('w2', r'$w_{t, d, n}$', x=x2, y=TOP-3, observed=True))
pgm.add_node(daft.Node('w3', r'$w_{t+1, d, n}$', x=x3, y=TOP-3, observed=True))
pgm.add_edge('z1', 'w1')
pgm.add_edge('z2', 'w2')
pgm.add_edge('z3', 'w3')

pgm.add_node(daft.Node('phi1', r'$\phi_{t-1, k}$', x=x1, y=TOP-5))
pgm.add_node(daft.Node('phi2', r'$\phi_{t, k}$', x=x2, y=TOP-5))
pgm.add_node(daft.Node('phi3', r'$\phi_{t+1, k}$', x=x3, y=TOP-5))
pgm.add_edge('phi1', 'phi2')
pgm.add_edge('phi2', 'phi3')
pgm.add_edge('phi1', 'w1')
pgm.add_edge('phi2', 'w2')
pgm.add_edge('phi3', 'w3')

w = .1
pgm.add_plate(daft.Plate([x_unit/2+w, TOP-5.5, 3*x_unit-2*w, 1-w], label='K', label_offset=(5, 10)))
# timeslices
pgm.add_plate(daft.Plate([x_unit/2+w, TOP-4.5, x_unit-2*w, 4], label=r'$D_{t-1}$', label_offset=(5, 10)))
pgm.add_plate(daft.Plate([x1+x_unit/2+w, TOP-4.5, x_unit-2*w, 4], label=r'$D_t$', label_offset=(5, 10)))
pgm.add_plate(daft.Plate([x2+x_unit/2+w, TOP-4.5, x_unit-2*w, 4], label=r'$D_{t+1}$', label_offset=(5, 10)))
# documents
pgm.add_plate(daft.Plate([x_unit/2+2*w, TOP-4, x_unit-4*w, 2.5], label=r'$N_{t-1,d}$', label_offset=(5, 10)))
pgm.add_plate(daft.Plate([x1+x_unit/2+2*w, TOP-4, x_unit-4*w, 2.5], label=r'$N_{t,d}$', label_offset=(5, 10)))
pgm.add_plate(daft.Plate([x2+x_unit/2+2*w, TOP-4, x_unit-4*w, 2.5], label=r'$N_{t+1,d}$', label_offset=(5, 10)))

pgm.render()
pgm.figure.show()
