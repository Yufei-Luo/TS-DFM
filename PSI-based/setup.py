from setuptools import setup

if __name__ == "__main__":
    setup()

#
#try: # for pip >= 10
#    from pip._internal.req import parse_requirements
#except ImportError: # for pip <= 9.0.3
#    from pip.req import parse_requirements
#
#try:  # pip >= 10
#    from pip._internal.download import PipSession
#except ImportError:  # pip <= 9.0.3
#    from pip.download import PipSession
#
#from setuptools import setup, find_packages
##from pip.req import parse_requirements
##from pip.download import PipSession
#
#install_reqs = parse_requirements("requirements.txt", session=PipSession())
#
#reqs = [str(ir.req) for ir in install_reqs] 
# 
# 
#setup(name='learnTS',
#      version='0.1',
#      url='https://gitlab.com/sunghwan.choi/learnts',
#      license='MIT',
#      author='Sunghwan Choi',
#      author_email='sunghwanchoi@kisti.re.kr',
#      description='TS structure learning',
#      packages=find_packages(include=['src']),
#      install_requires=reqs,
#      long_description=open('README.md').read(),
#      zip_safe=False,
#      )
