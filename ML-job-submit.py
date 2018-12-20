#!/usr/bin/env python
#
# Run script with run ID, site, filename containing LFNs from step 1

import sys
import time

from DIRAC.Core.Base import Script
Script.parseCommandLine()

from DIRAC.Core.Security.ProxyInfo import getProxyInfo
from DIRAC.Interfaces.API.Dirac import Dirac

# We construct the DIRAC Job Description Language as string in jdl:
jdl = ''

# Something descriptive for the name! Like 'FastRawMerging'.
jdl += 'JobName = "ML_test";\n'

if sys.argv[2] == 'sara':
 jdl += 'Site = "LCG.SARA-MATRIX.nl";\n'
 jdl += 'Platform = "EL7";\n'
 jdl += 'SmpGranularity = 4;\n'
 jdl += 'CPUNumber = 4;\n'
elif sys.argv[2] == 'manchester':
 jdl += 'Site = "LCG.UKI-NORTHGRID-MAN-HEP.uk";\n'
 jdl += 'GridCE = "ce01.tier2.hep.manchester.ac.uk";\n'
 jdl += 'Tags = "8Processors";\n'
 jdl += 'Platform = "EL7";\n'
elif sys.argv[2] == 'ral':
 jdl += 'Site = "LCG.RAL-LCG2.uk";\n'
 jdl += 'Tags = "8Processors";\n'
 jdl += 'GridCEQueue = "arc-ce03.gridpp.rl.ac.uk:2811/EL7";\n'
else:
 jdl += 'Tags = "8Processors";\n'
 jdl += 'Platform = "EL7";\n'

# The script you want to run. 
jdl += 'Executable = "grid_run.sh";\n'

# tarJob.sh will be run with these command line arguments
# %n is a counter increasing by one for each job in the list
# %s is the parameter taken from the list given in Parameters = { ... }
# %j is the unique DIRAC Job ID number
# something is just a value to show you can add other things too
jdl += 'Arguments = "ML_basic_test_run.py test_query_table_TGSSadded.pkl";\n'

jdl += 'InputSandbox = {"grid_run.sh", "ML_basic_test_run.py", "LFN:/skatelescope.eu//user/a/alex.clarke/AENEAS/test_query_table_TGSSadded.pkl", "LFN:/skatelescope.eu/user/a/alex.clarke/AENEAS/MLsing.simg", "prmon.tar.gz" };\n'

# Tell DIRAC where to get your big input data files from
# %s is the parameter taken from the list given in Parameters = { ... }
jdl += 'InputData = "";\n'

# Direct stdout and stderr to files
jdl += 'StdOutput = "StdOut";\n';
jdl += 'StdError = "StdErr";\n';

# Small files can be put in the output sandbox
jdl += 'OutputSandbox = {"StdOut", "StdErr", "prmon.txt", "run1.data", "run2.data", "run3.data", "run4.data"};\n'

# Files to be saved to your grid storage area in case they are large
# %j is the unique DIRAC Job ID number. 
# DIRAC looks for this output file in the working directory.
#jdl += 'OutputData = "LFN:/skatelescope.eu/user/r/rohini.joshi/PrefactorRuns/obs2/' + sys.argv[1] + '/cal2/cal_values.tar";\n'

# Give the OutputSE too if using OutputData:
# jdl += 'OutputSE = "UKI-NORTHGRID-MAN-HEP-disk";\n'	# storage in GridPP DIRAC
#jdl += 'OutputSE = "SARA-MATRIX-disk";\n'

# Tell DIRAC how many seconds your job might run for 
jdl += 'MaxCPUTime = 2000;\n'

# Create a unique Job Group for this set of jobs
try:
  diracUsername = getProxyInfo()['Value']['username']
except:
  print 'Failed to get DIRAC username. No proxy set up?'
  sys.exit(1)

#jobGroup = diracUsername + time.strftime('.%Y%m%d%H%M%S')
jobGroup = diracUsername + '.ML.' + sys.argv[1]
jdl += 'JobGroup = "' + jobGroup + '";\n'

print 'Will submit this DIRAC JDL:'
print '====='
print jdl
print '====='
print
# Submit the job(s)
print 'Attempting to submit job(s) in JobGroup ' + jobGroup
print
dirac = Dirac()
result = dirac.submit(jdl)
print
print '====='
print
print 'Submission Result: ',result
print
print '====='
print

if result['OK']:
  print 'Retrieve output with  dirac-wms-job-get-output --JobGroup ' + jobGroup
else:
  print 'There was a problem submitting your job(s) - see above!!!'
print
