
import geopandas 
import time
import matplotlib.pyplot as plt 

HCM_shapefile = geopandas.read_file('C:/Workspace/PythonForML/GIS/CSL_HCMC/Data/GIS/Population/population_HCMC/population_shapefile/Population_Ward_Level.shp')
# print('Head:\n', HCM_shapefile.head())
# HCM_shapefile.plot(column='Shape_Area', figsize=(16, 8))
# plt.show()
# keys = HCM_shapefile.keys()
# print(keys)
print('Họ và tên: Lại Chí Thiện')
print('MSSV: 20520309')
print('--------------------------------')
# Phường có diện tích lớn nhất là phường
print('Phường có diện tích lớn nhất là phường', HCM_shapefile.loc[HCM_shapefile["Shape_Area"].idxmax(),'Com_Name'], 'quận', HCM_shapefile.loc[HCM_shapefile["Shape_Area"].idxmax(),'Dist_Name'])
print('Phường có dân số 2019 cao nhất là phường', HCM_shapefile.loc[HCM_shapefile["Pop_2019"].idxmax(), 'Com_Name'], 'quận', HCM_shapefile.loc[HCM_shapefile["Pop_2019"].idxmax(), 'Dist_Name'])
print('Phường có diện tích nhỏ nhất là phường', HCM_shapefile.loc[HCM_shapefile["Shape_Area"].idxmin(), 'Com_Name'], 'quận', HCM_shapefile.loc[HCM_shapefile["Shape_Area"].idxmin(), 'Dist_Name'])
print('Phường có dân số 2019 thấp nhất là phường', HCM_shapefile.loc[HCM_shapefile["Pop_2019"].idxmin(), 'Com_Name'], 'quận', HCM_shapefile.loc[HCM_shapefile["Pop_2019"].idxmin(), 'Dist_Name'])
print('Phường có tốc độ tăng trưởng dân số nhanh nhất là phường', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] / HCM_shapefile["Pop_2009"]).idxmax(),'Com_Name'], 'quận', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] / HCM_shapefile["Pop_2009"]).idxmax(),'Dist_Name'])
print('Phường có tốc độ tăng trường dân số thấp nhất là phường', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] / HCM_shapefile["Pop_2009"]).idxmin(),'Com_Name'], 'quận', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] / HCM_shapefile["Pop_2009"]).idxmin(),'Dist_Name'])
print('Phường có biến động dân số nhanh nhất là phường', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] - HCM_shapefile["Pop_2009"]).idxmax(),'Com_Name'], 'quận', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] - HCM_shapefile["Pop_2009"]).idxmax(),'Dist_Name'])
print('Phường có biến động dân số nhanh nhất là phường', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] - HCM_shapefile["Pop_2009"]).idxmin(),'Com_Name'], 'quận', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] - HCM_shapefile["Pop_2009"]).idxmin(),'Dist_Name'])
print('Phường có mật độ dân số cao nhất là phường', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] / HCM_shapefile["Shape_Area"]).idxmax(),'Com_Name'], 'quận', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] / HCM_shapefile["Shape_Area"]).idxmax(),'Dist_Name'])
print('Phường có mật độ dân số thấp nhất là phường', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] / HCM_shapefile["Shape_Area"]).idxmin(),'Com_Name'], 'quận', HCM_shapefile.loc[(HCM_shapefile["Pop_2019"] / HCM_shapefile["Shape_Area"]).idxmin(),'Dist_Name'])