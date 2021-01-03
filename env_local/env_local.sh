#!/bin/bash 
set -xe

#export iplist=10.255.104.19,10.255.138.19,10.255.122.21
export iplist=`hostname -i` # ip of the local machine

#http_proxy
unset http_proxy
unset https_proxy

xml_parser=$(perldoc -l XML::Parser)
xml_parser_parent=$(dirname $xml_parser)
xml_parser_grandparent=$(dirname $xml_parser_parent)


lwp_user=$(perldoc -l LWP::UserAgent)
lwp_user_parent=$(dirname $lwp_user)
lwp_user_grandparent=$(dirname $lwp_user_parent)

export PERL5LIB=$xml_parser_grandparent:$lwp_user_grandparent

echo $PERL5LIB
