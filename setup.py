from setuptools import setup

setup(
    name='image_captioning',
    version='1.0',
    install_requires=[
        'torch',
        'transformers',
        'streamlit',
        'Pillow',
    ],
    scripts=['app.py'],
    entry_points={
        'console_scripts': [
            'image_captioning=app:main',
        ],
    },
)
