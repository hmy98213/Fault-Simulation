from dd import autoref as _bdd

bdd = _bdd.BDD()
bdd.declare('x', 'y', 'z')
u = bdd.add_expr(r'(x /\ y) \/ ~ z')
print(u.negated)
v = ~ u
print(v.negated)
bdd.collect_garbage()
bdd.dump('rooted.pdf', roots=[v])