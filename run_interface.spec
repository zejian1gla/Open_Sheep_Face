# -*- mode: python ; coding: utf-8 -*-
import sys
import os.path as osp
sys.setrecursionlimit(5000)

block_cipher = None

SETUP_DIR = r'C:\\Users\\zejian\\zejian_PhD_project\\packaged_pipeline\\'

a = Analysis(
    ['run_interface.py','test_on_image_or_firstframe.py'],
    pathex=[r'C:\Users\zejian\zejian_PhD_project\packaged_pipeline'],
    binaries=[(r"C:\Users\zejian\Sheep_demo\Bbox_pose_Pipeline\packaged_pipeline\venv\Lib\site-packages\scipy\.libs",'.')],
    datas=[(SETUP_DIR+'0-30.hkl','0-30.hkl'),(SETUP_DIR+'30-60.hkl','30-60.hkl'),(SETUP_DIR+'60-90.hkl','60-90.hkl'),
    (SETUP_DIR+'cascade_forest_training.log','cascade_forest_training.log'),(SETUP_DIR+'svm_models.pkl','svm_models.pkl'),
    (SETUP_DIR+'hopenet_weights/best_epoch_1.pk1','hopenet_weights/best_epoch_1.pk1'),(SETUP_DIR+'hopenet_weights/g_epoch_1.pk1','hopenet_weights/g_epoch_1.pk1'),
    (SETUP_DIR+'hopenet_weights/g_epoch_2.pk1','hopenet_weights/g_epoch_2.pk1'),(SETUP_DIR+'hopenet_weights/gg_epoch_2.pk1','hopenet_weights/gg_epoch_2.pk1'),
    (SETUP_DIR+'UI/Be_AI_logo.png','UI/Be_AI_logo.png'),(SETUP_DIR+'UI/lawnbackground.jpeg','UI/lawnbackground.jpeg'),
    (SETUP_DIR+'UI/lawnbackground.qrc','UI/lawnbackground.qrc'),(SETUP_DIR+'UI/Be_AI_logo.png','UI/Be_AI_logo.png'),
    (SETUP_DIR+'yolo_weights/1_2022_08_12best.pt','yolo_weights/1_2022_08_12best.pt'),(SETUP_DIR+'app/0-30.hkl','app/0-30.hkl'),
    (SETUP_DIR+'question_mark5.ico','question_mark5.ico')],
    hiddenimports=['pandas',"pytest",'pandas._libs','pandas._libs.tslibs.np_datetime','pandas._libs.tslibs.timedeltas',"sklearn.neighbors._typedefs","importlib_resources.trees"
             'pandas._libs.tslibs.nattype','scipy._lib','scipy._lib.messagestream','menpo','tensorflow','tensorflowjs','PyQt5','torch'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='run_interface',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run_interface',
)