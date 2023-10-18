#!/usr/bin/env python3

"""
This downloads the (almost) full e-obs dataset.
"""

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'insitu-gridded-observations-europe',
    {
        'format': 'zip',
        'variable': [
            'maximum_temperature', 'mean_temperature', 'precipitation_amount',
            'relative_humidity', 'wind_speed',
        ],
        'grid_resolution': '0.1deg',
        'period': 'full_period',
        'version': '27.0e',
        'product_type': 'ensemble_mean',
    },
    '/tmp/e-obs.zip')
