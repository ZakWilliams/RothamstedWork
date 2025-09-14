conversion_dict = {
    'Flow (l/s)' : 'Flow',
    'Pump (On/Off)' : 'Pump Status',
    #'Water Temperature Flume (oC)' : 'Water Flume T' Don't include this for now
    'Nitrite and Nitrate (mg/l)' : 'NitriteANDNitrate',
    'Ammonia (mg/l)' : 'Ammonia',
    'Ammonium (mg/l)' : 'Ammonium',
    'Conductivity (uS/cm)' : 'Conductivity',
    'Dissolved Oxygen (%)' : 'Dissolved Oxygen',
    'pH ()' : 'pH',
    #'Water Temperature Flow-cell (oC)' : 'Water Flow-cell T' Don't include this for now
    'Turbidity (FNU)' : 'Turbidity',
    'Fluorescent Dissolved Organic Matter (ug/l QSU)' : 'Fluor. Dis. Org. Matter',
    
    'Precipitation (mm)' : 'Precipitation',
    'Soil Temperature @ 15cm Depth (oC)' : 'Soil T @ 15cm',
    'Soil Moisture @ 10cm Depth (%)' : 'Soil Moisture @ 10cm'
}

SMP_data_types = ['Precipitation', 'Soil T @ 15cm', 'Soil Moisture @ 10cm']

FLOW_data_types = ['Flow', 'Pump Status',
    #'Water Flume T' Don't include this for now
    'NitriteANDNitrate',
    'Ammonia',
    'Ammonium',
    'Conductivity',
    'Dissolved Oxygen',
    'pH',
    #'Water Flow-cell T' Don't include this for now
    'Turbidity (FNU)',
    'Fluorescent Dissolved Organic Matter (ug/l QSU)',
]

# deposition chemicals - 'N', 'P2O5', 'K2O', 'SO3', 'MANURE', 'LIME
# runoff chemicals - 'NitriteANDNitrate', 'Ammonia', 'Ammonium', 'Conductivity', 'Dissolved Oxygen', 'pH', 'Turbidity (FNU)', 'Fluorescent Dissolved Organic Matter (ug/l QSU)',

units_dict = {
    'N' : 'kg',
    'P2O5' : 'kg',
    'K2O' : 'kg',
    'SO3' : 'kg',
    'MANURE' : 'kg',
    'LIME' : 'kg',
    'NitriteANDNitrate' : 'mg/l',
    'Ammonia' : 'mg/l',
    'Ammonium' : 'mg/l',
    'Conductivity' : 'uS/cm',
    'Dissolved Oxygen' : '%',
    'pH' : '',
    'Turbidity (FNU)' : 'FNU',
    'Fluorescent Dissolved Organic Matter (ug/l QSU)' : 'ug/l QSU',
}

LaTeX_dict = {
    'N' : r'Nitrogen',
    'P2O5' : r'P$_{2}$O$_{5}$',
    'K2O' : r'K$_{2}$O',
    'SO3' : r'SO$_{3}$',
    'MANURE' : r'Manure',
    'LIME' : r'Lime',
    'NitriteANDNitrate' : r'NO$_{2}^-$ & NO$_{3}^-$',
    'Ammonia' : r'NH$_{3}$',
    'Ammonium' : r'NH$_{4}^+$',
    'Conductivity' : r'Conductivity',
    'Dissolved Oxygen' : r'Dissolved Oxygen',
    'pH' : r'pH',
    'Turbidity (FNU)' : r'Turbidity',
    'Fluorescent Dissolved Organic Matter (ug/l QSU)' : r'Fluorescent Dissolved Organic Matter (ug/l QSU)',
}