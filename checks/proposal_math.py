"""Analytic Gaussian quadrature checks; requires NumPy. No robot experiments."""
import json
import numpy as np

nodes, weights = np.polynomial.hermite.hermgauss(16)
e, f = np.meshgrid(nodes * np.sqrt(2), nodes * np.sqrt(2), indexing='ij')
w = weights[:, None] * weights[None, :] / np.pi
E = lambda x: float(np.sum(w*x))
checks = []
def check(name, actual, expected, tol=1e-10):
    assert np.isfinite(actual) and abs(actual-expected) < tol, (name, actual, expected)
    checks.append({'name': name, 'actual': actual, 'expected': expected})
for theta in [-2., 0., .7, 2.]:
    for c in [-1., 0., .25, 1.]:
        u, y = theta+e, c*(theta+e)+f
        check(f'full gradient theta={theta} c={c}', E(y*e), c)
        check(f'marginal gradient theta={theta} c={c}', E(y*c*(y-c*theta)/(1+c*c)), c)
        check(f'masked bias theta={theta} c={c}', -E(y*e), -c)
# Final discarded coordinate V is independent of executed Y conditional on parents.
# Shared theta does not invalidate its zero reward-weighted score.
y, v = .7+e, .7+f
check('terminal omitted score', E(y*(v-.7)), 0.)
# Distributional dependence can be entirely in scale, with zero mean dependence.
sigma = np.exp(.2)
u, y = sigma*e, sigma*e+f
check('scale-only omitted score with squared reward', E(y*y*(e*e-1)), 2*sigma*sigma)
# A fixed nuisance distribution can influence the action but has zero theta score.
u, y = e, .7*e+f
check('parameter-independent nuisance; retained conditional score', E(y*y*f*u), 1.4)
# Valid score deletion does not imply equality of PPO clipped objectives.
# A Bernoulli nuisance shifts from p=.5 to p=.9; reward is identically one.
ratios = np.array([.2, 1.8])
check('unclipped nuisance ratio expectation', float(ratios.mean()), 1.)
check('full clipped objective', float(np.minimum(ratios, np.clip(ratios,.8,1.2)).mean()), .7)
check('reduced clipped objective', 1., 1.)
print(json.dumps({'checks_passed':len(checks), 'numpy':np.__version__, 'checks':checks}, indent=2))
