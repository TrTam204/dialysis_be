# Settings used ONLY for migration/CI testing against a scratch database.
# Mirrors dialysis_system.settings but points DATABASES at a disposable test DB.
import os

from .settings import *  # noqa: F401,F403

DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': os.getenv('TEST_DB_NAME', 'dialysis_migration_test'),
        'USER': os.getenv('DB_USER', 'postgres'),
        'PASSWORD': os.getenv('DB_PASSWORD', 'change-me'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}
