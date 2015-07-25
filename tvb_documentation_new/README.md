TVB Documentation in raw form
====================================

The use-case that we recommend for all contributors is the following:

- Register http://www.thevirtualbrain.org/register/ and get a TVB_Distribution package
- Test it in its original form, to make sure it's compatible with your environment
- Read carefully TVB-Distribution/docs/ContributorsManual.pdf
  [The raw form: RST for this PDF, as well as for the rest of TVB Tutorials can be found on current repository]
- Create an account for you on Github
- Fork TVB packages (scientific_library AND / OR framework_tvb) in your Github Account
- Use script TVB_Distribution/bin/contributors_setup.sh 
  (you will need to pass your Github fork URL when calling this script)
- You should now have a clone of your Github repo on your local disk
- You can continue to use all the startup scripts in TVB_Distribution and 
  you do not need to install Python or dependencies on your own.
- You can modify files (e.g. Add a monitor) in the folder just cloned from Github, 
  and use Git as a normal user for commits/branches
- Any change you do in the cloned folder, should be visible in the script or web interface of TVB 
  (but in most cases a restart of TVB is required for changes to become visible)
- For contributing to tvb documentation, you can independently fork and later push into "tvb_docs" repository.
