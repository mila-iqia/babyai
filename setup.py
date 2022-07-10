from setuptools import setup

setup(
    name='babyai',
    version='1.1.2',
    license='BSD 3-clause',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    packages=['babyai', 'babyai.levels', 'babyai.utils'],
    install_requires=[
        'gym>=0.24.1',
        'numpy>=1.17.0',
        "torch>=0.12",
        'blosc>=1.5.1',
        'einops>=0.4',
        'gym_minigrid @ https://github.com/saleml/gym-minigrid/archive/minigrid-no-warning.zip'
    ],
)
