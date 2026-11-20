from pathlib import Path
import hashlib,json,subprocess
task=Path('/data/codex/flagtree-conceal-option-cmake-20261120')
rows=[]
for variant,backend,option,expected,negative in [
 ('base','xpu','OFF',False,True),
 ('fixed','xpu',None,True,False),
 ('fixed','xpu','ON',True,False),
 ('fixed','xpu','OFF',False,False),
 ('fixed','nvidia','ON',False,False),
 ('fixed','nvidia','OFF',False,False),
 ('fixed','ascend','ON',False,False),
]:
 name=f'{variant}-{backend}-{option or "default"}'
 build=task/name
 assert not build.exists()
 args=['cmake','-S',str(task),'-B',str(build),'-G','Ninja',
       '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',f'-DVARIANT={variant}',
       f'-DFLAGTREE_BACKEND={backend}',f'-DEXPECT_CONCEAL={"ON" if expected else "OFF"}']
 if option is not None: args.append(f'-DTRITON_XPU_CONCEAL_IR={option}')
 proc=subprocess.run(args,capture_output=True,text=True)
 (task/(name+'.configure.log')).write_text(proc.stdout+proc.stderr)
 row={'variant':variant,'backend':backend,'option':option or 'default ON',
      'expected_conceal':expected,'negative_control':negative,'configure_exit':proc.returncode}
 if negative:
  assert proc.returncode!=0 and 'Unexpected concealment' in proc.stderr
 else:
  assert proc.returncode==0,proc.stdout+proc.stderr
  built=subprocess.run(['cmake','--build',str(build)],capture_output=True,text=True)
  (task/(name+'.build.log')).write_text(built.stdout+built.stderr)
  assert built.returncode==0,built.stdout+built.stderr
  commands=json.loads((build/'compile_commands.json').read_text())
  assert len(commands)==1
  command=commands[0]['command']
  assert ('-DTRITON_CONCEAL_IR=1' in command)==expected
  assert '-UTRITON_CONCEAL_IR' not in command
  row.update(build_exit=built.returncode,actual_compile_command=command,
             actual_target_definitions=(build/'actual-definitions.txt').read_text().strip())
 rows.append(row)
report={'status':'PASS','base_commit':'bb8a53cc427fb426b65e38bfdd00d0f4217af686',
        'scope':'Real current-main CMake target macro with compiled tiny target; not a full current-main compiler build',
        'full_compiler_build_validation':'Separate release-e6a8799c control and one-line49ce3849 build, see attached receipts',
        'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [task/'base.cmake',task/'fixed.cmake',task/'CMakeLists.txt',task/'probe.cpp']},
        'cmake_version':subprocess.check_output(['cmake','--version'],text=True).splitlines()[0],
        'rows':rows}
(task/'cmake-contract-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
