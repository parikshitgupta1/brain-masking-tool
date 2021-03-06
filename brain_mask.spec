# -*- mode: python -*-

block_cipher = None


a = Analysis(['brain_mask.py'],
             pathex=['/net/rc-fs-nfs/ifs/data/NoSync/FNNDSC-NR/neuro/labs/grantlab/users/alejandro.valdes/projects/brain-masking-tool'],
             binaries=[],
             datas=[('models/json_models/unet_model.json','models/json_models/'),
             ('models/weights/unet_weights.h5', 'models/weights/')],
             hiddenimports=['pywt._extensions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='brain_mask',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
