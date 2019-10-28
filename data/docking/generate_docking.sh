

while read line; do
    echo $line
    if [[ $line == *"id"* ]]; then
        continue
    fi
    pdbid=$(echo $line | cut -d"," -f2)
    dudeid=$(echo $line | cut -d"," -f3)

    if [[ ! -d ./../pdbbind/v2018/${pdbid} ]] | [[ -f ./${pdbid}/${pdbid}_docked_decoys_10.pdb ]]; then
        continue
    fi

    mkdir ./${pdbid}
    obabel ./../pdbbind/v2018/${pdbid}/${pdbid}_pocket.pdb -xr -O ./${pdbid}_pocket.pdbqt

    if [[ -f ./../dude/${dudeid}/actives/actives_10.pdb ]]; then
        for i in {1..10}
        do
            obabel ./../dude/${dudeid}/actives/actives_${i}.pdb -xr -O ./${pdbid}_actives_${i}.pdbqt
            obabel ./../dude/${dudeid}/decoys/decoys_${i}.pdb -xr -O ./${pdbid}_decoys_${i}.pdbqt

            ./smina.static -r ./${pdbid}_pocket.pdbqt -l ./${pdbid}_actives_${i}.pdbqt \
                --autobox_ligand ./${pdbid}_actives_${i}.pdbqt --num_modes 1 --autobox_add 8 --exhaustiveness 20 \
                -o ./${pdbid}_docked_actives_${i}.pdbqt
            ./smina.static -r ./${pdbid}_pocket.pdbqt -l ./${pdbid}_decoys_${i}.pdbqt \
                --autobox_ligand ./${pdbid}_decoys_${i}.pdbqt --num_modes 1 --autobox_add 8 --exhaustiveness 20 \
                -o ./${pdbid}_docked_decoys_${i}.pdbqt

            sed -i '/MODEL/d' ./${pdbid}_docked_actives_${i}.pdbqt
            sed -i '/ENDMDL/d' ./${pdbid}_docked_actives_${i}.pdbqt
            sed -i '/MODEL/d' ./${pdbid}_docked_decoys_${i}.pdbqt
            sed -i '/ENDMDL/d' ./${pdbid}_docked_decoys_${i}.pdbqt

            obabel ./${pdbid}_docked_actives_${i}.pdbqt -xr -O ./${pdbid}/${pdbid}_docked_actives_${i}.pdb
            obabel ./${pdbid}_docked_decoys_${i}.pdbqt -xr -O ./${pdbid}/${pdbid}_docked_decoys_${i}.pdb
        done
    fi
    ln -s ${PWD}/../pdbbind/v2018/${pdbid}/${pdbid}_protein.pdb ${PWD}/${pdbid}
    ln -s ${PWD}/../pdbbind/v2018/${pdbid}/${pdbid}_pocket.pdb ${PWD}/${pdbid}
    ln -s ${PWD}/../pdbbind/v2018/${pdbid}/${pdbid}_ligand.pdb ${PWD}/${pdbid}

    rm ./*.pdbqt

done <./../dude2pdb.csv
