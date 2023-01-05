#!/usr/bin/env python3
import subprocess
import argparse
import os
import sys
import configparser
import platform


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def read_conf(filename):
    config = configparser.ConfigParser(interpolation=None, allow_no_value=True, delimiters=('=',))
    with open(filename) as f:
        config.read_file(f)
    return config


def to_dirs(keys):
    return ['"{}"'.format(d.replace("\\", "\\\\")) for d in keys]


def main():
    parser = argparse.ArgumentParser('Bootstraps this project using conan')
    parser.add_argument('--build_type', type=str, required=False)
    args = parser.parse_args()

    subprocess.check_call([
        'pip3',
        'install',
        'conan',
    ])

    try:
        subprocess.check_call([
            'conan',
            'profile',
            'new',
            'default',
            '--detect',
        ], stderr=subprocess.PIPE)
    except BaseException as e:
        pass

    if platform.system() == 'Linux':
        subprocess.check_call([
            'conan',
            'profile',
            'update',
            'settings.compiler.libcxx=libstdc++11',
            'default',
        ])

    types = [args.build_type] if args.build_type else ['Debug', 'Release']
    for build_type in types:

        subprocess.check_call([
            'conan',
            'install',
            '--build=missing',
            THIS_DIR,
            '-s',
            'build_type={}'.format(build_type),
        ])

        config = read_conf(os.path.join(THIS_DIR, 'conanbuildinfo.txt'))
        includedirs = to_dirs(config['includedirs'].keys())
        libdirs = to_dirs(config['libdirs'].keys())

        cmake_dir = os.path.join(THIS_DIR, 'cmake')
        if not os.path.exists(cmake_dir):
            os.mkdir(cmake_dir)
        with open(os.path.join(cmake_dir, '{}.cmake'.format(build_type)), 'w') as f:
            f.write('include_directories({})\n'.format(' '.join(includedirs)))
            f.write('link_directories({})\n'.format(' '.join(libdirs)))

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
