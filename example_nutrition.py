 #%pip install -r requirements.txt

import pandas as pd
import cfe.regression as rgsn

Malawi_Data = '1MHl6EPsSoUXLQWdkiVob97eX2CepZnWxLIB6u-PzkNo'
Unutrition = '1yFWlP5N7Aowaj6t2roRSFFUC50aFD-RLBGfzGtqLl0w'

import pandas as pd
import numpy as np
from eep153_tools.sheets import read_sheets

# Change 'Malawi_Data' to key of your own sheet in Sheets, above
x = read_sheets(Malawi_Data,sheet='Food Expenditures (2019-20)')
x = x.set_index(['i','t','m','j']).squeeze()


# Now prices
p = read_sheets(Malawi_Data,sheet='Food Prices (2019-20)').set_index(['t','m','j','u'])
#pd.read_csv('Malawi_Food_Prices(2019-20).csv')


# Compute medians of prices for particular time, place and unit
p = p.groupby(['t','m','j','u']).median()

# Just keep metric units
p = p.xs('kg',level="u").squeeze().unstack('j')

r = rgsn.read_pickle('malawi_estimates.rgsn')

# Get intersection of goods we have prices *and* expenditures for:
jidx = p.columns.intersection(r.beta.index)

# Drop prices for goods we don't have expenditures for
p = p[jidx].T


# Household characteristics
d = read_sheets(Malawi_Data,sheet="Household Characteristics")
d.columns.name = 'k'

# Fill blanks with zeros
d = d.replace(np.nan,0)

# Expenditures x may have duplicate columns
x = x.T.groupby(['i','t','m','j']).sum()
x = x.replace(0,np.nan) # Replace zeros with missing

# Take logs of expenditures; call this y
y = np.log(x)

d.set_index(['i','t','m'],inplace=True)

r = rgsn.Regression(y=y,d=d)

fct = pd.read_csv('Malawi_FCT (1).csv')

fct = fct.set_index('j')
fct.columns.name = 'n'

fct = fct.apply(lambda x: pd.to_numeric(x,errors='coerce'))

fct.index = fct.index.str.lower().str.strip()

rdi = read_sheets(Unutrition, sheet='RDI')

rdi = rdi.set_index('n')
rdi.columns.name = 'k'

rdi = rdi.apply(lambda x: pd.to_numeric(x,errors='coerce'))

# Assumes you've already set this up e.g., in Project 3
r = rgsn.read_pickle('malawi_estimates.rgsn')

# Reference prices chosen from a particular time; average across place.
# These are prices per kilogram:
p.index = [pr.lower() for pr in p.index]
pbar = p.loc[[food.lower() for food in r.beta.index]].mean(axis=1).fillna(1) # Only use prices for goods we can estimate
pbar.index = [s.capitalize() for s in pbar.index]

import numpy as np

xhat = r.predicted_expenditures()


# Total food expenditures per household
xbar = xhat.groupby(['i','t','m']).sum()



# Reference budget
xref = xbar.quantile(0.5)  # Household at 0.5 quantile is median

qhat = (xhat.unstack('j')/pbar).dropna(how='all')

# Drop missing columns
qhat = qhat.loc[:,qhat.count()>0]


def my_prices(j,p0,p=pbar):
    """
    Change price of jth good to p0, holding other prices fixed at p.
    """
    p = p.copy()
    p.loc[j] = p0
    return p

import matplotlib.pyplot as plt

use = 'Maize ufa mgaiwa (normal flour)'  # Good we want demand curve for

# Vary prices from 50% to 200% of reference.
scale = np.linspace(.5,2,20)

# Demand for Matoke for household at median budget
plt.plot([r.demands(xref,my_prices(use,pbar[use]*s,pbar))[use] for s in scale],scale)

# Demand for Matoke for household at 25% percentile
plt.plot([r.demands(xbar.quantile(0.25),my_prices(use,pbar[use]*s,pbar))[use] for s in scale],scale)

# Demand for Matoke for household at 75% percentile
plt.plot([r.demands(xbar.quantile(0.75),my_prices(use,pbar[use]*s,pbar))[use] for s in scale],scale)

plt.ylabel(f"Price (relative to base of {pbar[use]:.2f})")
plt.xlabel(f"Quantities of {use} Demanded")

fig,ax = plt.subplots()

scale = np.geomspace(.01,10,50)

ax.plot(np.log(scale*xref),[r.expenditures(s*xref,pbar)/(s*xref) for s in scale])
ax.set_xlabel(f'log budget (relative to base of {xref:.0f})')
ax.set_ylabel(f'Expenditure share')
ax.set_title('Engel Curves')


# Create a new FCT and vector of consumption that only share rows in common:
display(fct)
fct0,c0 = fct.align(qhat.T,axis=0,join='inner')
print(fct0.index)

### Diagnostic



# The @ operator means matrix multiply
N = fct0.T@c0

N  #NB: Malawi quantities are for previous 7 days

#def nutrient_demand(x,p):
 #   c = r.demands(x,p)
#
  #  fct_clean = fct[~fct.index.duplicated()]
  #  c_clean = c[~c.index.duplicated()] if isinstance(c, pd.DataFrame) else c
   # 
    #fct0, c0 = fct_clean.align(c_clean, axis=0, join='inner')
    #N = fct0.T @ c0
#
 #   N = N.loc[~N.index.duplicated()]
    
  #  return N

def nutrient_demand(x, p):
    c = r.demands(x, p)

    # Standardize indices
    c.index = c.index.str.lower().str.strip()
    c = c[~c.index.duplicated()]

    fct_clean = fct.copy()
    fct_clean.index = fct_clean.index.str.lower().str.strip()
    
    # Drop foods missing nutrient info
    fct_clean = fct_clean.dropna(subset=UseNutrients)
    fct_clean = fct_clean[~fct_clean.index.duplicated()]

    # Align on shared foods
    fct0, c0 = fct_clean.align(c, axis=0, join='inner')

    if fct0.empty or c0.empty:
        print("No valid nutrient-demand matches found.")
        return pd.DataFrame()

    N = fct0.T @ c0
    N = N.loc[~N.index.duplicated()]
    return N

import numpy as np
import matplotlib.pyplot as plt

UseNutrients = ['Protein','Energy','Iron','Vitamin A']

### Diagnostics

c = r.demands(xref, pbar)
c.index = c.index.str.lower().str.strip()
c = c[~c.index.duplicated()]

fct_clean = fct.copy()
fct_clean.index = fct_clean.index.str.lower().str.strip()
fct_clean = fct_clean[~fct_clean.index.duplicated()]

fct0, c0 = fct_clean.align(c, axis=0, join='inner')

print("Number of goods in alignment:", len(fct0))
print("Total quantity demanded (sum):", c0.sum())
print("Total nutrients available:\n", (fct0.T @ c0)[UseNutrients])


#print("fct index sample:", [k for k in fct.index])
#sample_c = r.demands(xref, pbar)
#print("r.demands index sample:", [k for k in c.index])
print("Common goods:", fct.index.intersection(c.index))

test_N = nutrient_demand(xref, pbar)
print("Nutrient demand:\n", test_N[UseNutrients])

missing_nutrients = fct.loc[
    fct.index.isin([
        'avocado', 'banana', 'bean, brown', 'beef', 'biscuits', 'bread',
        'buns, scones', 'cabbage', 'cassava tubers', 'chinese cabbage',
        'chips (vendor)', 'cooking oil', 'eggs', 'fresh fish (medium variety)',
        'fresh fish (small variety)', 'fresh milk', 'fruit juice', 'goat',
        'green maize', 'groundnut flour', 'irish potato',
        'maize ufa mgaiwa (normal flour)', 'maize ufa refined (fine flour)',
        'mandazi, doughnut (vendor)', 'onion', 'orange sweet potato', 'popcorn',
        'powdered milk', 'rice', 'salt', 'samosa (vendor)',
        'smoked fish (medium variety)', 'smoked fish (small variety)',
        'soyabean flour', 'spaghetti, macaroni, pasta', 'spices', 'sugar',
        'sugar cane', 'sun dried fish (large variety)',
        'sun dried fish (medium variety)', 'sun dried fish (small variety)',
        'sweets, candy, chocolates', 'tanaposi/rape', 'tea', 'thobwa', 'tomato',
        'white sweet potato', 'yoghurt'
    ]),
    UseNutrients
]

print("Missing nutrient entries:\n", missing_nutrients.isna().sum())

## end diagnostics

X = np.linspace(xref / 5, xref * 5, 50)

df = pd.concat({
    myx: np.log(nutrient_demand(myx, pbar).clip(lower=1e-6))[UseNutrients]
    for myx in X
}, axis=1).T

ax = df.plot()
ax.set_xlabel('log budget')
ax.set_ylabel('log nutrient')
ax.set_title('Log Nutrient Content vs Budget')

#print([k for k in pbar.index])

pbar.index = [k.lower() for k in pbar.index]

USE_GOOD = 'maize ufa mgaiwa (normal flour)'

scale = np.geomspace(0.01, 10, 50)

ndf = pd.DataFrame({
    s: np.log(nutrient_demand(xref / 2, my_prices(USE_GOOD, pbar[USE_GOOD] * s)).clip(lower=1e-6))[UseNutrients]
    for s in scale
}).T

ax = ndf.plot()
ax.set_xlabel('log price (relative scale) post change')
ax.set_ylabel('log nutrient')
#ax.set_title(f'Log Nutrient vs Price of {USE_GOOD}')

# In first round, averaged over households and villages
dbar = r.d[rdi.columns].mean()

# This matrix product gives minimum nutrient requirements for
# the average household
hh_rdi = rdi@dbar

hh_rdi

def nutrient_adequacy_ratio(x,p,d,rdi=rdi,days=7):
    hh_rdi = rdi.replace('',0)@d*days

    return nutrient_demand(x,p)/hh_rdi

X = np.geomspace(.01*xref,2*xref,100)

pd.DataFrame({x:np.log(nutrient_adequacy_ratio(x,pbar,dbar))[UseNutrients] for x in X}).T.plot()
plt.legend(UseNutrients)
plt.xlabel('budget')
plt.ylabel('log nutrient adequacy ratio (post change)')
plt.axhline(0)
plt.axvline(xref)

scale = np.geomspace(0.01, 2, 50)

ndf = pd.DataFrame({
    s * pbar[USE_GOOD]: np.log(
        nutrient_adequacy_ratio(xref / 4, my_prices(USE_GOOD, pbar[USE_GOOD] * s), dbar).clip(lower=1e-6)
    )[UseNutrients]
    for s in scale
}).T

fig, ax = plt.subplots()
ax.plot(ndf['Energy'], ndf.index)
ax.axhline(pbar[USE_GOOD], linestyle='--', color='gray')
ax.axvline(0, linestyle='--', color='gray')
ax.set_ylabel('Price')
ax.set_xlabel('log nutrient adequacy ratio (Iron) original ')
ax.set_title('Iron Adequacy vs Price')
