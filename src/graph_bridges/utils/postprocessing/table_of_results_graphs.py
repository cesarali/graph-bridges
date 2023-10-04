"""

    from pprint import pprint
    from graph_bridges import results_path

    table_of_results = GraphTablesOfResults()

    #===================================================================
    # READ EXPERIMENT AND CHANGE TABLE
    #===================================================================

    _,_,experiment_dir = get_experiment_dir(results_path,
                                            experiment_name="graphs",
                                            experiment_type="",
                                            experiment_indentifier="lobster_to_efficient_one_1685024983")
    configs,metrics,models,results = table_of_results.read_experiment_dir(experiment_dir)

    print(table_of_results.config_to_dataset_name(configs))
    print(table_of_results.config_to_method_name(configs))

    dataset_id,method_id,metrics_in_file,missing_in_file = table_of_results.experiment_dir_to_table(experiment_dir,False,True)
    pprint(table_of_results.create_pandas())

    stuff = table_of_results.experiment_dir_to_model(None,experiment_dir)
    dataset_name, method_name, metrics_in_file, missing_in_file, graph_diffusion_model = stuff

    #===================================================================
    # DESIGN OF EXPERIMENTS
    #===================================================================

    from graph_bridges import project_path
    from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig

    config = CTDDConfig()

    config = table_of_results.dataset_name_to_config("Community",config)
    table_of_results.run_config(config)

    #pprint(config)


"""