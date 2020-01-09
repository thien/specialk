#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

use warnings;
use strict;

my $language = "en";
my $PENN = 0;

while (@ARGV) {
    $_ = shift;
    /^-b$/ && ($| = 1, next); # not buffered (flush each line)
    /^-l$/ && ($language = shift, next);
    /^[^\-]/ && ($language = $_, next);
  	/^-penn$/ && ($PENN = 1, next);
}

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
    if ($PENN == 0) {
      s/\`/\'/g;
      s/\'\'/ \" /g;
    }

    s/„/\"/g;
    s/“/\"/g;
    s/”/\"/g;
    s/–/-/g;
    s/—/ - /g; s/ +/ /g;
    s/´/\'/g;
    s/([a-z])‘([a-z])/$1\'$2/gi;
    s/([a-z])’([a-z])/$1\'$2/gi;
    s/‘/\'/g;
    s/‚/\'/g;
    s/’/\"/g;
    s/''/\"/g;
    s/´´/\"/g;
    s/…/.../g;
    # French quotes
    s/ « / \"/g;
    s/« /\"/g;
    s/«/\"/g;
    s/ » /\" /g;
    s/ »/\"/g;
    s/»/\"/g;
    # handle pseudo-spaces
    s/ \%/\%/g;
    s/nº /nº /g;
    s/ :/:/g;
    s/ ºC/ ºC/g;
    s/ cm/ cm/g;
    s/ \?/\?/g;
    s/ \!/\!/g;
    s/ ;/;/g;
    s/, /, /g; s/ +/ /g;

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
    
    # (Modication)
    s/([0-9]) ([0-9])/$1 $2/gi;
    s/Ã§/ç/g;
    s/àª/ê/g;
    s/Ã©/é/g;
    s/Ã€/ì/g;
    s/Ã¨/è/g;
    s/Å/œ/g;
    s/Ã /à /g; 
    s/Ã®/î/g;
    s/Ã¢/â/g;
    s/Ã¹/ù/g;
    s/Ã"/â/g;
    s/Ã‰/ð/g;
    s/Ã /À/g;
    s/Ã /Â/g;
    s/Ã /Ä/g;
    s/Ã /È/g;
    s/Ã /É/g;
    s/Ã /Ê/g;
    s/Ã /Ë/g;
    s/Ã /Î/g;
    s/Ã /Ï/g;
    s/Ã /Ô/g;
    s/Å /Œ/g;
    s/Ã /Ù/g;
    s/Ã /Û/g;
    s/Ã /Ü/g;
    s/Å¸/Ÿ/g;
    s/Ã/à/g;
    s/Ã¢/â/g;
    s/Ã¤/ä/g;
    s/Ã¨/è/g;
    s/Ã©/é/g;
    s/Ãª/ê/g;
    s/Ã«/ë/g;
    s/Ã®/î/g;
    s/Ã¯/ï/g;
    s/Ã´/ô/g;
    s/Å /œ/g;
    s/Ã¹/ù/g;
    s/Ã»/û/g;
    s/Ã¼/ü/g;
    s/Ã¿/ÿ/g;


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


    if ($language eq "de" || $language eq "es" || $language eq "cz" || $language eq "cs" || $language eq "fr") {
	s/(\d) (\d)/$1,$2/g;
    }
    else {
	s/(\d) (\d)/$1.$2/g;
    }
    print $_;
}
