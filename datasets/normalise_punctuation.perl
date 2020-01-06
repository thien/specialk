#!/usr/bin/perl -w
# source: www.statmt.org/wmt11/normalize-punctuation.perl

use strict;

my ($language) = @ARGV;

while(<STDIN>) {
    s/\r//g;
    # remove extra spaces
    s/\(/ \(/g;
    s/\)/\) /g; s/ +/ /g;
    s/\) ([\.\!\:\?\;\,])/\)$1/g;
    s/\( /\(/g;
    s/ \)/\)/g;
    s/(\d) \%/$1\%/g;
    s/ :/:/g;
    s/ ;/;/g;
    # normalize unicode punctuation
    s/â€ž/\"/g;
    s/â€œ/\"/g;
    s/â€/\"/g;
    s/â€“/-/g;
    s/â€”/ - /g; s/ +/ /g;
    s/Â´/\'/g;
    s/([a-z])â€˜([a-z])/$1\'$2/gi;
    s/([a-z])â€™([a-z])/$1\'$2/gi;
    s/â€˜/\"/g;
    s/â€š/\"/g;
    s/â€™/\"/g;
    s/''/\"/g;
    s/Â´Â´/\"/g;
    s/â€¦/.../g;
    # French quotes
    s/Â Â«Â / \"/g;
    s/Â«Â /\"/g;
    s/Â«/\"/g;
    s/Â Â»Â /\" /g;
    s/Â Â»/\"/g;
    s/Â»/\"/g;
    # handle pseudo-spaces
    s/Â \%/\%/g;
    s/nÂºÂ /nÂº /g;
    s/Â :/:/g;
    s/Â ÂºC/ ÂºC/g;
    s/Â cm/ cm/g;
    s/Â \?/\?/g;
    s/Â \!/\!/g;
    s/Â ;/;/g;
    s/,Â /, /g; s/ +/ /g;

    # English "quotation," followed by comma, style
    if ($language eq "en") {
	s/\"([,\.]+)/$1\"/g;
    }
    # Czech is confused
    elsif ($language eq "cs" || $language eq "cz") {
    }
    # German/Spanish/French "quotation", followed by comma, style
    else {
	s/,\"/\",/g;	
	s/(\.+)\"(\s*[^<])/\"$1$2/g; # don't fix period at end of sentence
    }

    print STDERR $_ if /ï»¿/;

    if ($language eq "de" || $language eq "es" || $language eq "cz" || $language eq "cs" || $language eq "fr") {
	s/(\d)Â (\d)/$1,$2/g;
    }
    else {
	s/(\d)Â (\d)/$1.$2/g;
    }
    print $_;
}

