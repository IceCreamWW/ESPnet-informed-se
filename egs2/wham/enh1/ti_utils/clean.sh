cp text text_bak
sed -i "s/[^A-Z0-9_' <>]//g" text
sed -i "s/'SINGLEQUOTE/SINGLEQUOTE/g" text
sed -i 's/<NOISE>//g' text
