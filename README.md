# Fractional Differentiation on Time Series
As described in Advances of Machine Learning by Marcos Prado

## SP500 returns

<p align="center">
  <img src="doc/frac_diff_sp500.png">
</p>

## F(X) = X and its (frac) derivatives/antiderivates

<p align="center">
  <img src="doc/fx_animation.gif">
</p>

## Get Started

```
git clone git@github.com:philipperemy/fractional-differentiation-time-series.git && cd fractional-differentiation-time-series
virtualenv -p python3 venv
source venv/bin/activate
pip install . --upgrade
python frac_diff_sp500.py
```


References:
- https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
- https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
- https://en.wikipedia.org/wiki/Fractional_calculus
