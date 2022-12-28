#!/usr/bin/env python3
import subprocess
import argparse
import os
import sys
import configparser


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def read_conf(filename):
    config = configparser.ConfigParser(interpolation=None, allow_no_value=True)
    with open(filename) as f:
        config.read_file(f)
    return config


def get_packages():
    config = read_conf(os.path.join(THIS_DIR, 'conanfile.txt'))
    packages = {}
    for pkg in config['requires'].keys():
        elements = pkg.split('/')
        packages[elements[0]] = elements[1]
    return packages


def find_package_dir(name, version, build_type):
    base_dir = os.path.join(os.environ['HOME'], 
                            '.conan',
                            'data',
                             name,
                             version,
                             '_',
                             '_',
                             'package')
    for dir in os.listdir(base_dir):
         info = read_conf(os.path.join(base_dir, dir, 'conaninfo.txt'))
         if 'build_type' not in info['settings']:
             # header-only lib
             return os.path.join(base_dir, dir)
         if info['settings']['build_type'] == build_type:
             return os.path.join(base_dir, dir)
    assert False, "No package dir found for: {}, {}, {}".format(name, version, build_type)


def main():
    parser = argparse.ArgumentParser('Bootstraps this project using conan')
    parser.add_argument('--build_type', type=str, required=True)
    args = parser.parse_args()
    subprocess.check_call([
        'pip3',
        'install',
        'conan',
    ])
    subprocess.check_call([
        'conan',
        'install',
        '--build=missing',
        THIS_DIR,
        '-s',
        'build_type={}'.format(args.build_type),
    ])
    packages = get_packages()
    cxxflags = []
    ldflags = []
    for name, version in packages.items():
        dir = find_package_dir(name, version, args.build_type)
        cxxflags.append('-I{}/include'.format(dir))
        ldflags.append('-L{}/lib'.format(dir))
    with open(os.path.join(THIS_DIR, '{}.cmake'.format(args.build_type)), 'w') as f:
        f.write('set(CMAKE_CXX_FLAGS "{}")\n'.format(' '.join(cxxflags)))
        f.write('set(CMAKE_EXE_LINKER_FLAGS "{}")\n'.format(' '.join(ldflags)))
    garbage = [
        'conan.lock',
        'conanbuildinfo.txt',
        'conaninfo.txt',
        'graph_info.json',
    ]
    for g in garbage:
        if os.path.exists(g):
            os.remove(g)


if __name__ == '__main__':
    main()
