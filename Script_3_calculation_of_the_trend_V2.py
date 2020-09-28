#Editor : Logothetis Stavros-Andreas Logothetis 
#Laboratory of Atmospheric Physics, University of Patras
#Project: Trend analysis of Dust and Aerosol Optical Depth in global scale

#Main idea of the script : calculate the trends for the geometric means for all grids with include more than 5 years data

import xarray as xr
import numpy as np
import glob
import os 

months = range(1,181)

x = []
for month in months:
        x_d = round(month/12,3)
        x.append(x_d)
    
Xt = np.array(x)

directory = os.getcwd()

filenames = glob.glob('*.nc')
filenames_time = glob.glob(r''+directory+'/time/*.nc')

x_array = xr.open_mfdataset(filenames_time, combine='by_coords')

x_array = x_array.get(['anomalies_arithmetic_mean_DOD'])
x_array['anomalies_geometric_mean_DOD'] = x_array['anomalies_arithmetic_mean_DOD']
x_array = x_array.drop('anomalies_arithmetic_mean_DOD')

x_array.coords['Time'] = Xt

y_array = xr.open_mfdataset(filenames, combine='by_coords')
y_array.coords['Time'] = Xt

n = y_array.notnull().sum(dim='Time')

y_array = y_array.where(n>60)
x_array = x_array.where(n>60)

xmean = x_array.mean(axis=0)
ymean = y_array.mean(axis=0)
xstd  = x_array.std(axis=0)
ystd  = y_array.std(axis=0)

#4. Compute covariance along time axis
cov  =  np.sum((x_array - xmean)*(y_array - ymean), axis=0)/(n)

#5. Compute correlation along time axis
cor   = cov/(xstd*ystd)

#6. Compute regression slope and intercept:
slope = cov/(xstd**2)

intercept = ymean - xmean*slope  

slope.to_netcdf(r''+directory+'/Results/slope.nc')
intercept.to_netcdf(r''+directory+'/Results/intercept.nc')

#7. Compute P-value and standard error
#Compute t-statistics
tstats = cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
stderr = slope/tstats

from scipy.stats import t
pval   = t.sf(tstats.anomalies_geometric_mean_DOD, n.anomalies_geometric_mean_DOD-2)*2
pval   = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)
pval.to_netcdf(r''+directory+'/pvalue_DOD.nc')
