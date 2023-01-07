from setuptools import setup, find_packages

def main():
    try:
        requirements_file   =   open('./requirements.txt','rt',encoding='UTF-8')
        requirements_content    =   requirements_file.read()
        requirements_file.close()
    except IOError or OSError as e:
        print(e)
        print('Could not open requirements.txt, exiting.')
        return 1

    requirements    =   requirements_content.split('\n')
    requirements    =   [i for i in requirements if len(i) > 0]

    setup(
        name        =   'LegalEval2023_OrleansUniversity',
        version     =   '0.1.5',
        packages    =   find_packages(include=requirements)
    )
    return 0

if __name__ == '__main__':
    main_result =   main()
    print(f'setup.py finished with status {main_result}')
