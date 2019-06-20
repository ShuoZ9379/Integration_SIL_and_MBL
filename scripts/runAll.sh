#!/bin/bash

# Environments
environments=(RoboschoolReacher-v1 RoboschoolInvertedDoublePendulum-v1 RoboschoolHopper-v1 RoboschoolWalker2d-v1 RoboschoolHalfCheetah-v1 RoboschoolAnt-v1 RoboschoolHumanoid-v1 RoboschoolHumanoidFlagrunHarder-v1 RoboschoolAtlasForwardWalk-v1)
state_dim=6

nitr=1000
horizon=1000
discount=0.99
batch_size=10000
timesteps=10000000

dryrun=false
minseed=0
maxseed=9

# Run experiments in all environments
for n in $(seq 0 8);
do
    environment=${environments[$n]}

    #
    # TRPO
    #
    epsilon=0.01
    for compatible in 0 1;
    do
        for seed in $(seq $minseed $maxseed);
        do
            if [ "$n" = 3 ] ; then
                for entbonus in 0.001 0.005 0.01 0.05 0.1;
                do
                    cmd="python -m baselines.n_copos.run_roboschool --num-timesteps ${timesteps} --timesteps_per_episode ${batch_size} --env-id ${environment} --seed ${seed} --retrace 0 --trpo 1 --entropy_bonus ${entbonus} --epsilon ${epsilon} --n_policy 1 --compatible ${compatible}"
                    if [ "$dryrun" = true ] ; then
                        echo "$cmd"
                    else
                        sbatch runExperiment.sh "$cmd"
                    fi
                    sleep 0.1
                done
            fi
            entbonus=0.0

            for npolicy in 1 4;
            do
                if [ "$npolicy" = 1 ] ; then
                    retrace=0
                else
                    retrace=1
                fi
                
                cmd="python -m baselines.n_copos.run_roboschool --num-timesteps ${timesteps} --timesteps_per_episode ${batch_size} --env-id ${environment} --seed ${seed} --retrace ${retrace} --trpo 1 --entropy_bonus ${entbonus} --epsilon ${epsilon} --n_policy ${npolicy} --compatible ${compatible}"
                if [ "$dryrun" = true ] ; then
                    echo "$cmd"
                else
                    sbatch runExperiment.sh "$cmd"
                fi
                sleep 0.1
            done
        done
    done

    #
    # COPOS
    #
    epsilon=0.01
    compatible=1
    for beta in -1 0.01;
    do
        for seed in $(seq $minseed $maxseed);
        do
            for npolicy in 1 4;
            do
                if [ "$npolicy" = 1 ] ; then
                    retrace=0
                else
                    retrace=1
                fi
                
                cmd="python -m baselines.n_copos.run_roboschool --num-timesteps ${timesteps} --timesteps_per_episode ${batch_size} --env-id ${environment} --seed ${seed} --retrace ${retrace} --trpo 0 --entropy_bonus 0.0 --epsilon ${epsilon} --n_policy ${npolicy} --beta ${beta} --compatible ${compatible}"
                if [ "$dryrun" = true ] ; then
                    echo "$cmd"
                else
                    sbatch runExperiment.sh "$cmd"
                fi
                sleep 0.1
            done
        done
    done

    #
    # DDPG
    #
    for seed in $(seq $minseed $maxseed);
    do
        cmd="python -m baselines.ddpg.main --nb-epochs 1000 --nb-rollout-steps 1000 --nb-epoch-cycles 10 --num-timesteps 10000000 --env-id ${environment} --seed ${seed}"

        if [ "$dryrun" = true ] ; then
            echo "$cmd"
        else
	    sbatch runExperiment.sh "$cmd"
        fi
	sleep 0.1
    done

    #
    # PPO
    #
    epsilon=0.01
    for seed in $(seq $minseed $maxseed);
    do
        if [ "$n" = 3 ] ; then
            for entbonus in 0.001 0.005 0.01 0.05 0.1;
            do
                cmd="python -m baselines.ppo1.run_roboschool --num-timesteps ${timesteps} --timesteps_per_episode ${batch_size} --entropy-coeff ${entbonus} --env-id ${environment} --seed ${seed}"
                
                if [ "$dryrun" = true ] ; then
                    echo "$cmd"
                else
                    sbatch runExperiment.sh "$cmd"
                fi
                sleep 0.1
            done
        fi
        entbonus=0.0
        
        cmd="python -m baselines.ppo1.run_roboschool --num-timesteps ${timesteps} --timesteps_per_episode ${batch_size} --entropy-coeff ${entbonus} --env-id ${environment} --seed ${seed}"
        
        if [ "$dryrun" = true ] ; then
            echo "$cmd"
        else
            sbatch runExperiment.sh "$cmd"
        fi
        sleep 0.1
    done
done
