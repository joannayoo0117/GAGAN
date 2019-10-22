# Downloading both general and refined set
wget http://pdbbind.org.cn/download/pdbbind_v2018_other_PL.tar.gz -P ./data
wget http://pdbbind.org.cn/download/pdbbind_v2018_refined.tar.gz -P ./data

# Unzipping the data
mkdir ./data/pdbbind
tar -xzvf ./data/pdbbind_v2018_other_PL.tar.gz -C ./data/pdbbind
tar -xzvf ./data/pdbbind_v2018_refined.tar.gz -C ./data/pdbbind

mv ./data/pdbbind/refined-set/* ./data/pdbbind/v2018-other-PL/
mv ./data/pdbbind/v2018-other-PL/ ./data/pdbbind/v2018


# Using openbabel, convert .sdf to .pdb
# Remove water from .pdb
# Remove unused files
cd ./data/pdbbind/v2018
for pdbid in *; do
    if [ "${pdbid}" == "index" ] || [ "${pdbid}" == "readme" ]
    then
        echo "Skipping ${pdbid} file"
        continue
    else
        echo ${pdbid}
        obabel ${pdbid}/${pdbid}_ligand.sdf -xr -O ${pdbid}/${pdbid}_ligand.pdb > /dev/null 2>&1
        mv ${pdbid}/${pdbid}_protein.pdb  ${pdbid}/${pdbid}_protein_hoh.pdb
        grep -v 'HOH' ${pdbid}/${pdbid}_protein_hoh.pdb >  ${pdbid}/${pdbid}_protein.pdb
        mv ${pdbid}/${pdbid}_ligand.pdb  ${pdbid}/${pdbid}_ligand_hoh.pdb
        grep -v 'HOH' ${pdbid}/${pdbid}_ligand_hoh.pdb >  ${pdbid}/${pdbid}_ligand.pdb
        rm ${pdbid}/${pdbid}_ligand_hoh.pdb  ${pdbid}/${pdbid}_protein_hoh.pdb
        rm ${pdbid}/${pdbid}_ligand.sdf  ${pdbid}/${pdbid}_ligand.mol2
    fi
done

cd ../../../
rm -rf ./data/pdbbind/refined-set
rm ./data/pdbbind_v2018_other_PL.tar.gz
rm ./data/pdbbind_v2018_refined.tar.gz


