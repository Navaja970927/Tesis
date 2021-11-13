import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'Credit Card Fraud Detection'
AUTHOR = 'Odeynis Valdés Suárez'
AUTHOR_EMAIL = 'odeynisvaldes@gmail.com'
URL = 'https://github.com/Navajas970927/Tesis'

DESCRIPTION = 'Librería para crear modelos DL y realizar pruebas y comparaciones'
LONG_DESCRIPTION = (HERE / "Readme.md").read_text(encoding='utf-8')
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    'scikit-learn', 'xgboost', 'imblearn', 'pandas', 'keras', 'tensorflow', 'numpy', 'seaborn', 'openpyxl', 'pylab'
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    include_package_data=True
)
