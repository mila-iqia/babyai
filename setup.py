from setuptools import setup

setup(
    name='babyai',
    version='1.1',
    license='BSD 3-clause',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    packages=['babyai', 'babyai.levels', 'babyai.utils'],
    install_requires=[
        'gym==0.23', # before introduction of 'truncated' extra output... (previously: >=0.9.6')
        'numpy',
        "torch>=0.4.1",
        'blosc>=1.5.1',
        'gym-minigrid==1.1', # before gym-minigrid became Minigrid (previously 'gym_minigrid @ https://github.com/maximecb/gym-minigrid/archive/master.zip')
    ],
)
