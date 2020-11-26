"""
Simulator backends provide pluggable numerical implementations,
allowing for different implementations or strategies.

- Follows roughly the DAO pattern which isolates the service layer from database API
  - current simulator+datatypes is the compute "service"
  - backends back the compute service
  - service layer passes traited types into backend
  - backend handles the translation to a particular compute api (numpy, tf)
- Isolates array creation, tracing, assist switching in float32
- 'Service' layer receives closures or generator
- Backend specifies preferred types and array creation routines
- Components can then be associated with a component, e.g.
  - nest component uses nest backend
  - field componetn uses shtns backend
- Isolates array creation, tracing, assist switching in float32
- 'Service' layer receives closures or generator
- Backend specifies preferred types and array creation routines
- Components can then be associated with a component, e.g.
  - nest component uses nest backend
  - field componetn uses shtns backend
- Multibackend-multicomponents need conversions done

"""


class BaseBackend:
    "Type tag for backends."

    @staticmethod
    def default(self):
        "Get default backend."
        # TODO later allow for configuration
        from .ref import ReferenceBackend
        return ReferenceBackend()
