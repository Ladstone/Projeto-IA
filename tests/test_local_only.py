"""Regression tests: no cloud/provider and prediction is input-dependent."""
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.vision import VisionService
from PIL import Image
import numpy as np

client=TestClient(app)

def test_ui_has_no_api_key_field():
    html=client.get('/').text.lower()
    assert 'type="password"' not in html
    assert 'aiprovider' not in html

def test_analyze_has_only_local_mode():
    example=(Path(__file__).parent/'fixtures'/'rice.jpg').read_bytes()
    result=client.post('/api/analyze',files={'file':('rice.jpg',example,'image/jpeg')})
    assert result.status_code==200,result.text
    data=result.json()
    assert data['mode']=='local'
    assert len(data['vision_debug'])==28
    assert data['analysis_quality']=='experimental'
    assert all(it['class_id'] in range(1,22) for it in data['detected'])

def test_different_examples_generate_different_segmentation_masks():
    root=Path(__file__).parents[1]
    model=VisionService(root/'models/unified_21class.pt')
    rice=np.asarray(model.predict(Image.open(Path(__file__).parent/'fixtures'/'rice.jpg'))[0])
    bean=np.asarray(model.predict(Image.open(Path(__file__).parent/'fixtures'/'bean.jpg'))[0])
    assert rice.shape==bean.shape
    assert not np.array_equal(rice,bean) # Proves result isn't a fixed hardcoded menu.
