import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key")
DEBUG = os.environ.get("DJANGO_DEBUG", "false").lower() == "true"
ALLOWED_HOSTS = [h.strip() for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",") if h.strip()]

# rag-api is only ever called server-side from here, never from the browser —
# no CORS setup needed anywhere in this stack.
RAG_API_URL = os.environ.get("RAG_API_URL", "http://rag-api:8000")

# Minimal app set: this is a stateless proxy + static file server, no auth,
# no forms, no database — keeps the image and attack surface small.
INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "ragproxy",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # angular-app/dist/chat-ui/browser holds the built index.html + assets,
        # copied here by the Dockerfile's build stage.
        "DIRS": [BASE_DIR / "static-app"],
        "APP_DIRS": False,
        "OPTIONS": {"context_processors": []},
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# No database — nothing in this service is stateful.
DATABASES = {}

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static-app"]
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

USE_TZ = True
