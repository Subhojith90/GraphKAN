import pandas as pd
import pytest
from graphkan_amo_nqs.plotting.main_figures import reject_smoke_only

def test_plot_rejects_smoke():
    with pytest.raises(ValueError):
        reject_smoke_only(pd.DataFrame({"profile":["quick_ci"]}), "fig_paper.pdf", allow_smoke=False)
