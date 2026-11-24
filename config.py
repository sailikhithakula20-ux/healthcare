import os
from datetime import timedelta
from dotenv import load_dotenv
from sqlalchemy.pool import NullPool

# Load environment variables from .env file
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))


def normalize_database_url(url):
    """
    Normalize PostgreSQL database URLs for SQLAlchemy.
    - If the user provides postgresql+psycopg://, preserve it (Psycopg 3).
    - If the user provides postgresql+psycopg2://, preserve it (Psycopg 2).
    - If postgres:// or bare postgresql:// is provided, normalize to postgresql+psycopg:// (or psycopg2).
    - Enforces sslmode=require for cloud PostgreSQL (Supabase).
    """
    if not url:
        return url

    clean_url = url.strip()

    # Normalize protocol: if standard postgres:// or postgresql:// without explicit driver
    if clean_url.startswith('postgres://'):
        clean_url = clean_url.replace('postgres://', 'postgresql+psycopg://', 1)
    elif clean_url.startswith('postgresql://'):
        clean_url = clean_url.replace('postgresql://', 'postgresql+psycopg://', 1)

    # Ensure SSL is required for remote PostgreSQL connections if not already specified
    if 'postgresql' in clean_url and 'sslmode' not in clean_url:
        delimiter = '&' if '?' in clean_url else '?'
        clean_url = f"{clean_url}{delimiter}sslmode=require"

    return clean_url


def is_postgres_pooler(url):
    """Detect if the connection string targets a PostgreSQL instance or connection pooler."""
    if not url:
        return False
    return 'postgres' in url.lower()


def get_database_uri(env_var_name='DATABASE_URL'):
    """Resolve and normalize database URI dynamically from environment."""
    raw_url = os.environ.get(env_var_name, '').strip()
    if raw_url:
        return normalize_database_url(raw_url)
    return f"sqlite:///{os.path.join(basedir, 'instance', 'health_portal.db')}"


def get_engine_options(uri):
    """Return appropriate engine options based on target database dialect."""
    if is_postgres_pooler(uri):
        return {
            'poolclass': NullPool,
            'connect_args': {
                'sslmode': 'require'
            }
        }
    return {
        'pool_recycle': 280,
        'pool_pre_ping': True,
    }


class Config:
    """Base application configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-dev-secret-key-replace-in-prod')
    APP_NAME = os.environ.get('APP_NAME', 'Preventive Health AI Portal')
    
    # SQLAlchemy configuration: SQLite default for local, Supabase PostgreSQL when DATABASE_URL is set
    SQLALCHEMY_DATABASE_URI = get_database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = get_engine_options(SQLALCHEMY_DATABASE_URI)

    # Session & Security
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600

    # Generative AI Configuration
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

    # Machine Learning Model Artifacts Directory
    ML_MODELS_DIR = os.path.join(basedir, 'ml', 'saved_models')
    RAG_DATA_DIR = os.path.join(basedir, 'rag')


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False


class TestingConfig(Config):
    """Testing environment configuration."""
    DEBUG = False
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SQLALCHEMY_ENGINE_OPTIONS = {}
    WTF_CSRF_ENABLED = False
    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    """Production environment configuration for Vercel / Cloud."""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True

    # In production on Vercel, DATABASE_URL is mandatory for data persistence
    @classmethod
    def init_app(cls, app):
        raw_url = os.environ.get('DATABASE_URL', '').strip()
        if not raw_url and os.environ.get('VERCEL'):
            raise RuntimeError(
                "CRITICAL: DATABASE_URL environment variable is missing on Vercel deployment. "
                "Please configure your Supabase PostgreSQL connection string in Vercel Project Settings."
            )

    SQLALCHEMY_DATABASE_URI = get_database_uri()
    SQLALCHEMY_ENGINE_OPTIONS = get_engine_options(SQLALCHEMY_DATABASE_URI)


def get_config(config_name=None):
    """Return configured Config class with dynamically resolved environment parameters."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    cfg = config_by_name.get(config_name, DevelopmentConfig)
    # Refresh dynamic database URI and engine options at runtime
    if config_name != 'testing':
        cfg.SQLALCHEMY_DATABASE_URI = get_database_uri()
        cfg.SQLALCHEMY_ENGINE_OPTIONS = get_engine_options(cfg.SQLALCHEMY_DATABASE_URI)
    return cfg


config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
