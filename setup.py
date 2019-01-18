from setuptools import setup

setup(
    name='babyai',
    version='0.0.2',
    license='BSD 3-clause',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    packages=['babyai', 'babyai.levels', 'babyai.utils'],
    install_requires=[
        'gym>=0.9.6',
        'numpy==1.15.4', # Temporary: fix numpy version because of bug introduced in 1.16
        'pyqt5>=5.10.1',
        "torch>=0.4.1",
        'gym_minigrid',
        'blosc>=1.5.1'
    ],
    dependency_links=[
        'git+https://github.com/maximecb/gym-minigrid.git#egg=gym_minigrid-0',
    ]
)
