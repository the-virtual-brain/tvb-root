#! /usr/bin/env python
# encoding: utf-8

APPNAME = 'cpuid'
VERSION = '4.0.0'

import waflib.extras.wurf_options


def options(opt):

    opt.load('wurf_common_tools')


def resolve(ctx):

    import waflib.extras.wurf_dependency_resolve as resolve

    ctx.load('wurf_common_tools')

    ctx.add_dependency(resolve.ResolveVersion(
        name='waf-tools',
        git_repository='github.com/steinwurf/waf-tools.git',
        major=3))

    ctx.add_dependency(resolve.ResolveVersion(
        name='platform',
        git_repository='github.com/steinwurf/platform.git',
        major=2))

    # Internal dependencies
    if ctx.is_toplevel():

        ctx.add_dependency(resolve.ResolveVersion(
            name='boost',
            git_repository='github.com/steinwurf/boost.git',
            major=2))

        ctx.add_dependency(resolve.ResolveVersion(
            name='gtest',
            git_repository='github.com/steinwurf/gtest.git',
            major=3))


def configure(conf):

    conf.load("wurf_common_tools")


def build(bld):

    bld.load("wurf_common_tools")

    bld.env.append_unique(
        'DEFINES_STEINWURF_VERSION',
        'STEINWURF_CPUID_VERSION="{}"'.format(VERSION))

    bld.recurse('src/cpuid')

    if bld.is_toplevel():

        # Only build tests when executed from the top-level wscript,
        # i.e. not when included as a dependency
        bld.recurse('test')
        bld.recurse('examples/print_cpuinfo')
