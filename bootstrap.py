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
    base_dir = os.path.join(os.path.expanduser('~'),
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
    parser.parse_args()
    for build_type in ['Debug', 'Release']:
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
            'build_type={}'.format(build_type),
        ])
        packages = get_packages()
        cxxflags = []
        ldflags = []
        for name, version in packages.items():
            dir = find_package_dir(name, version, build_type)
            cxxflags.append('"{}/include"'.format(dir))
            ldflags.append('"{}/lib"'.format(dir))
        with open(os.path.join(THIS_DIR, '{}.cmake'.format(build_type)), 'w') as f:
            f.write('include_directories({})\n'.format(' '.join(cxxflags)))
            f.write('link_directories({})\n'.format(' '.join(ldflags)))
        garbage = [
            'conan.lock',
            'conanbuildinfo.txt',
            'conaninfo.txt',
            'graph_info.json',
        ]
        for g in garbage:
            fg = os.path.join(THIS_DIR, g)
            if os.path.exists(fg):
                os.remove(fg)


if __name__ == '__main__':
    main()
