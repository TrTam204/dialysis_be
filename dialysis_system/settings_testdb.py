# Settings used ONLY for migration/CI testing against a scratch database.
# Mirrors dialysis_system.settings but points DATABASES at a disposable test DB.
from .settings import *  # noqa: F401,F403

DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': 'dialysis_migration_test',
        'USER': 'postgres',
        'PASSWORD': '050104',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
