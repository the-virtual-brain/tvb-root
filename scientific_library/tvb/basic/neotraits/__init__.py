"""
Neotraits is a framework that lets you declare class attributes with type checking and introspection abilities.
The public api is in the neotraits.api module
"""

# note: the api is in it's own module.
# This is a bit less convenient.
# But it prevents the annoying situation where
# importing neotraits.some_module will run __init__
# and that one will import most modules in order to provide the api.
# That is not ideal because circular imports become likely and over-importing can slow startup.
