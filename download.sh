wget http://dude.docking.org/db/subsets/all/all.tar.gz -P ./data

mkdir ./data/dude
tar -xzvf ./data/all.tar.gz -C ./data/dude
mv ./data/dude/all/* ./data/dude
rm -rf ./data/dude/all

cd ./data/dude
for d in */; do
    cd ${d}
    echo "Processing $d"
    mkdir actives
    mkdir decoys
    gunzip actives_final.mol2.gz
    gunzip decoys_final.mol2.gz

    if [[ -f ./actives_final.mol2  ]] && [[ -f ./decoys_final.mol2  ]]; then
        obabel actives_final.mol2 -O actives/actives_.pdb --separate -m  > /dev/null 2>&1
        num_actives=$(ls -l actives/*.pdb | wc -l)
        echo "Found $num_actives actives in $d"
        delete_from=$(grep -n '@<TRIPOS>MOLECULE'  decoys_final.mol2 | head -${num_actives} | tail -1 | cut -d":" -f1)
        echo "Deleting from line $delete_from from decoy file"
        head -${delete_from} decoys_final.mol2 > decoys_smaller.mol2
        obabel decoys_smaller.mol2 -O decoys/decoys_.pdb --separate -m  > /dev/null 2>&1
        obabel crystal_ligand.mol2 -xr -O ligand.pdb  > /dev/null 2>&1
    fi

    rm *.ism
    rm *.mol2
    rm *.sdf.gz

    echo "Done processing $d"
    echo "======================"
    cd ../
done

rm -rf ./data/all.tar.gz

# Downloading PDBBind (both general and refined set)
wget http://pdbbind.org.cn/download/pdbbind_v2018_other_PL.tar.gz -P ./data
wget http://pdbbind.org.cn/download/pdbbind_v2018_refined.tar.gz -P ./data

# Unzipping the data
mkdir ./data/pdbbind
tar -xzvf ./data/pdbbind_v2018_other_PL.tar.gz -C ./data/pdbbind
tar -xzvf ./data/pdbbind_v2018_refined.tar.gz -C ./data/pdbbind

mv ./data/pdbbind/refined-set/index ./data/pdbbind/v2018-other-PL/refined-index
mv ./data/pdbbind/refined-set/readme ./data/pdbbind/v2018-other-PL/refined-readme
mv ./data/pdbbind/refined-set/* ./data/pdbbind/v2018-other-PL/
mv ./data/pdbbind/v2018-other-PL/ ./data/pdbbind/v2018


# Using openbabel, convert .sdf to .pdb
# Remove water from .pdb
# Remove unused files
cd ./data/pdbbind/v2018
for pdbid in *; do
    if [ "${pdbid}" == *"index"* ] || [ "${pdbid}" == *"readme"* ]
    then
        echo "Skipping ${pdbid} file"
        continue
    else
        echo ${pdbid}
        obabel ${pdbid}/${pdbid}_ligand.sdf -xr -O ${pdbid}/${pdbid}_ligand.pdb > /dev/null 2>&1
        mv ${pdbid}/${pdbid}_protein.pdb  ${pdbid}/${pdbid}_protein_hoh.pdb
        grep -v 'HOH' ${pdbid}/${pdbid}_protein_hoh.pdb >  ${pdbid}/${pdbid}_protein.pdb
        mv ${pdbid}/${pdbid}_pocket.pdb  ${pdbid}/${pdbid}_pocket_hoh.pdb
        grep -v 'HOH' ${pdbid}/${pdbid}_pocket_hoh.pdb >  ${pdbid}/${pdbid}_pocket.pdb
        mv ${pdbid}/${pdbid}_ligand.pdb  ${pdbid}/${pdbid}_ligand_hoh.pdb
        grep -v 'HOH' ${pdbid}/${pdbid}_ligand_hoh.pdb >  ${pdbid}/${pdbid}_ligand.pdb
        rm ${pdbid}/${pdbid}_ligand_hoh.pdb  ${pdbid}/${pdbid}_protein_hoh.pdb ${pdbid}/${pdbid}_pocket_hoh.pdb
        rm ${pdbid}/${pdbid}_ligand.sdf  ${pdbid}/${pdbid}_ligand.mol2
    fi
done

cd ../../../
rm -rf ./data/pdbbind/refined-set
rm ./data/pdbbind_v2018_other_PL.tar.gz
rm ./data/pdbbind_v2018_refined.tar.gz

echo "Done downloading data!"
