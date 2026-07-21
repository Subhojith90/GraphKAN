from graphkan_amo_nqs.cli import main

def test_cli_help():
    assert main([]) == 0
