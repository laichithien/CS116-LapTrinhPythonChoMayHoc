import geopandas
import time 
import matplotlib.pyplot as plt 

HCM_shapefile = geopandas.read_file('./GIS/CSL_HCMC/Data/GIS/Population/population_HCMC/population_shapefile/Population_Ward_Level.shp')

# plt.text('Head:\n', HCM_shapefile.head, s="")
# plt.text('Key:\n', HCM_shapefile.keys)
# print('Head:\n', HCM_shapefile.head())
# HCM_shapefile.plot(column = 'Shape_Area', figsize=(16,8))
# plt.show()
# print('Keys:\n', HCM_shapefile.keys())

# print('Top 5 quận có mật độ dân số tăng nhanh nhất:\n')
# print(HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] / HCM_shapefile["Shape_Area"])])
print(HCM_shapefile.keys())
