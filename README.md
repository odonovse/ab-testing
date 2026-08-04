# Lumantic-AB Package

This is an open-source AB-testing package that will provide both
Frequentist and Bayesian methodologies. We will be building it
piecemeal from scratch and iterating towards greater complexity
and flexibility.



## Installation

Can be done with the following command in Terminal after you have
cloned the repo.

```bash
python3 -m pip install -e lumen-ab/
```

and then in python it will be as simple as:

```python
from lumantic_ab import t_test_summary
```



## Future Direction

For now the repo only offers the **most basic** Frequentist functions
for a Welch's T-test (*so we allow unequal variances across groups*),
with the following assumptions built in:

- there are only two-groups (classic AB test)
- the groups have equal sizes (50:50 split)
- the assignment column is boolean (*true = treatment, false =*
  *control*)

The plans for the future will be to gradually relax each of these
assumptions, while incorporating other necessary features for an
AB-testing:  SRM checks, Winsorisation, CUPED adjustments/DiD, 
and fixed effects; as well as Bayesian functions.

If you have any thoughts or ideas, or want to collaborate please
feel free to reach out: 
- email: odonovse@tcd.ie
- telegram: @odonovse
- X: @DataNerdAlways
- LinkedIn: https://www.linkedin.com/in/seamus-o-donovan/
