import pytest
from nma.simulation import generate_shared_private

@pytest.fixture
def aligned():
    return generate_shared_private(n_samples=100,random_state=21).aligned
