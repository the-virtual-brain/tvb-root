from tvb.basic.neotraits.ex import TraitAttributeError
from tvb.simulator.monitors import Raw,NArray,Float
from tvb.simulator.history import BaseHistory, Dim, NDArray
from types import MethodType
import numpy


class Model_with_proxy():
    '''
    Minimum variables and functions for creating a model with proxy
    '''
    _id_proxy = [] # identifier of the proxy node
    _update = False # if the proxy node have define by the integrator
    _proxy_value = None # the value of the proxy node for current computation

    def copy_inst(self, model_without_proxy):
        '''
        Copy the value of an instance without proxy
        :param model_without_proxy: model without proxy
        '''
        for key, value in vars(model_without_proxy).items():
            try:
               setattr(self, key, value)
            except TraitAttributeError:
                # variable final don't need to copy
                pass
        self.configure()

    def set_id_proxy(self, id_proxy):
        '''
        define the list of the different proxy
        :param id_proxy: list with the identifier of the proxy node
        '''
        self._id_proxy = id_proxy

    def update_proxy(self, data):
        '''
        the new data for the next computation
        :param data: valeu for the proxy node
        '''
        self._update = True
        self._proxy_value = data

    def coupling_variable(self):
        '''
        function in order to have access to the variable of the  coupling variable
        :return: firing rate of the model
        '''
        return self._coupling_variable

class HistoryProxy(BaseHistory):
    "History implementation for saving proxy data."

    nb_proxy = Dim()  # number of proxy
    dim_1 = Dim()  # dimension one
    dt = NDArray((dim_1, dim_1), 'f')  # integration time of the simulator
    current_step = NDArray((dim_1, dim_1), 'f', read_only=False)  # the current time step of the simulation
    id_proxy = NDArray((nb_proxy,), 'i')  # the identification of proxy node
    buffer = NDArray(('n_time', 'n_cvar', 'n_node', 'n_mode'), 'f8', read_only=False)  # the buffer of value of the proxy

    def __init__(self, time_synchronize, dt, id_proxy, cvars, n_mode):
        '''
        initialisation of the history for saving proxy value
        :param time_synchronize: time between two receiving value
        :param dt: time of integration
        :param id_proxy: list of the proxy node
        :param cvars: the number of coupling variable
        :param n_mode: the number of mode
        '''
        #TODO send warning for bad size delay
        dim = max(1, (len(id_proxy)))
        size_history = numpy.zeros((dim, dim), dtype=numpy.int)  # size of number of node
        size_history[0, 0] = int(time_synchronize / dt) + 1  # size of the number of saving step
        super(HistoryProxy, self).__init__(None, size_history, cvars, n_mode)
        self.dim_1 = 1
        self.dt = numpy.array([dt])
        self.nb_proxy = id_proxy.shape[0]
        self.id_proxy = id_proxy

    def update(self, step, data):
        '''
        update the history with new value
        :param step: the step of the update value
        :param data: the data for the proxy node
        '''
        if self.id_proxy.size != 0:
            step_n = data[0] / self.dt[0] - step  # the index of the buffer
            if any(step_n > self.n_time):  # check if there are not too much data
                raise Exception('ERROR too early')
            if any(numpy.rint(step_n).astype(int) < 0.0):  # check if it's not missing value
                raise Exception('ERROR too late')
            indice = numpy.expand_dims(numpy.rint(step_n + step).astype(int) % self.n_time, 1)
            if indice.size != numpy.unique(indice).size:  # check if the index is correct
                raise Exception('ERRROR two times are the same')
            self.buffer[indice, :,numpy.arange(0,self.id_proxy.shape[0]), :] = data[1]

    def next_step(self):
        '''
        :return: return the next step
        '''
        return self.buffer[(int(self.current_step)) % self.n_time]

    def update_current_step(self, step):
        '''
        update the current step of the simulator
        :param step: current step
        :return:
        '''
        self.current_step = numpy.array([step])

############################################ Modify Reduced Wong Wang #############################################
# Modification of the Reduced Wong Wang in order to accept proxy value and to retutn the firing rate of the differnet node

from tvb.simulator.models.wong_wang import ReducedWongWang
from numba import guvectorize, float64

@guvectorize([(float64[:],)*12], '(n),(m)' + ',()'*8 + '->(n),(n)', nopython=True)
def _numba_dfun(S, c, a, b, d, g, ts, w, j, io, dx,h):
    "Gufunc for reduced Wong-Wang model equations.(modification for saving the firing rate h)"
    x = w[0]*j[0]*S[0] + io[0] + j[0]*c[0]
    h[0] = (a[0]*x - b[0]) / (1 - numpy.exp(-d[0]*(a[0]*x - b[0])))
    dx[0] = - (S[0] / ts[0]) + (1.0 - S[0]) * h[0] * g[0]

@guvectorize([(float64[:],)*5], '(n),(m)' + ',()'*2 + '->(n)', nopython=True)
def _numba_dfun_proxy(S, h, g, ts, dx):
    "Gufunc for reduced Wong-Wang model equations for proxy node."
    dx[0] = - (S[0] / ts[0]) + (1.0 - S[0]) * h[0] * g[0]

class ReducedWongWang_proxy(ReducedWongWang,Model_with_proxy):
    '''
    modify class in order to take in count proxy firing rate and to monitor the firing rate
    '''
    def __init__(self):
        super(ReducedWongWang,self).__init__()

    def dfun(self, x, c, local_coupling=0.0):
            # same has tvb implementation
            x_ = x.reshape(x.shape[:-1]).T
            c_ = c.reshape(c.shape[:-1]).T + local_coupling * x[0]
            deriv, H = _numba_dfun(x_, c_, self.a, self.b, self.d, self.gamma,
                                   self.tau_s, self.w, self.J_N, self.I_o)
            # compute the part for the proxy node if there are data
            if self._update and self._id_proxy.size != 0:
                # take the parameter for the proxy nodes
                tau_s = self.tau_s if self.tau_s.shape[0] == 1 else self.tau_s[self._id_proxy]
                gamma = self.gamma if self.gamma.shape[0] == 1 else self.gamma[self._id_proxy]
                # compute the derivation
                deriv[self._id_proxy] = _numba_dfun_proxy(x_[self._id_proxy], self._proxy_value, gamma, tau_s)
                # replace the firing rate by the true firing rate
                H[self._id_proxy] = self._proxy_value
            if self._update:
                # replace the firing rate by the true firing rate
                self._coupling_variable = H
                self._update = False
            return deriv.T[..., numpy.newaxis]
#######################################################################################

class Interface_co_simulation(Raw):
    id_proxy = NArray(
        dtype=numpy.int,
        label = "Identifier of proxies",
        )
    time_synchronize = Float(
        label="simulated time between receiving the value of the proxy",
        )

    _model_with_proxy=None

    def __init__(self,model_with_proxy=None,**kwargs):
        super(Interface_co_simulation, self).__init__(**kwargs)
        self._model_with_proxy=model_with_proxy

    def config_for_sim(self,simulator):
        #configuration of all monitor
        super(Interface_co_simulation, self).config_for_sim(simulator)
        self._id_node =  numpy.where(numpy.logical_not(numpy.isin(numpy.arange(0,simulator.number_of_nodes,1), self.id_proxy)))[0]
        self._nb_step_time = numpy.int(self.time_synchronize/simulator.integrator.dt)

        # create the model with proxy
        if hasattr(simulator.model, '_id_proxy'):
            self._model_with_proxy = simulator.model.__class__
        else:
            if self._model_with_proxy is None:
                model_class = simulator.model.__class__
                class model_proxy(model_class,Model_with_proxy):
                    '''
                        modify class in order to take in count proxy and to monitor the coupling variable
                    '''
                    def __init__(self):
                        super(model_class,self).__init__()

                    def dfun(self, x, c, local_coupling=0.0):
                        if self._update:
                            self._coupling_variable = x[self.cvar, :]
                            if self._id_proxy.size != 0:
                                x[self.cvar,self._id_proxy]=self._proxy_value
                            self._update=False
                        deriv = super(model_proxy,self).dfun(x, c, local_coupling)
                        return deriv
                self._model_with_proxy = model_proxy
            else :
                if not hasattr(self._model_with_proxy(), '_id_proxy'):
                    raise Exception("ERROR type of model doesn't accept proxy")  #avoid bad type of class
            new_model = self._model_with_proxy()
            new_model.copy_inst(simulator.model)
            simulator.model = new_model
        self.model = simulator.model
        self.model.set_id_proxy(self.id_proxy)

        ######## WARNING:Change the instance simulator for taking in count the proxy ########
        # overwrite of the simulator for update the proxy value
        class Simulator_proxy (type(simulator)):
            #Modify the call method of the simulator in order to update the proxy value
            def __call__(self, simulation_length=None, random_state=None, proxy_data=None):
                if hasattr(self.integrator, 'history_proxy') and proxy_data is not None:
                    self.integrator.update_proxy_history(self.current_step + 1, proxy_data)
                return super(type(simulator),self).__call__(simulation_length=simulation_length, random_state=random_state)

        #change the class of the simulator
        simulator.__class__ = Simulator_proxy

        # overwrite of the method _loop_compute_node_coupling of the simulator :
        # this method is the first method call in the integration loop.
        # This overwriting add the update of the current step for the integrator
        original_method = simulator._loop_compute_node_coupling
        def coupling(step):
            return original_method(step)
        self.coupling=coupling
        def _loop_compute_node_coupling(self,step):
            ''''
            see the simulator for this method
            '''
            self.integrator.history_proxy.update_current_step(step)
            return original_method(step)
        simulator._loop_compute_node_coupling = MethodType(_loop_compute_node_coupling,simulator)

        ######## WARNING:Change the instance of integrator for taking in count the proxy ########
        # Modify the Integrator for manage the proxy node :

        ## Add an history for saving the value from external input
        simulator.integrator.history_proxy = HistoryProxy(self.time_synchronize,simulator.integrator.dt,self.id_proxy,simulator.model.cvar,simulator.model.number_of_modes)
        def update_proxy_history(self, step, proxy_data ):
            '''
            update the history with the new value
            :param step: the current step
            :param proxy_data: the value of proxy node
            '''
            self.history_proxy.update(step,proxy_data)
        # add a method for update the history of the integrator
        simulator.integrator.update_proxy_history = MethodType(update_proxy_history, simulator.integrator)

        ## use the data in history to compute next step (overwrite of the method scheme)
        simulator.integrator._scheme_original_ = simulator.integrator.scheme
        simulator.integrator.interface_co_simulation = self # access to the model method (I am not sure to be the best way)
        def scheme(self, X, dfun, coupling, local_coupling, stimulus):
            self.interface_co_simulation.model.update_proxy(self.history_proxy.next_step())
            return self._scheme_original_(X, dfun, coupling, local_coupling, stimulus)
        simulator.integrator.scheme = MethodType(scheme,simulator.integrator)

    def sample(self, step, state):
        '''
        record fo the monitor in order to send result of TVB
        :param step: current step
        :param state: the state of all the node and also the value of the proxy
        :return:
        '''
        self.step = step
        time = numpy.empty((2,),dtype=object)
        time[:] = [step * self.dt, (step+self._nb_step_time)*self.dt]
        result = numpy.empty((2,),dtype=object)
        result[:] = [numpy.expand_dims(self.model.coupling_variable(),0),self.coupling(step+self._nb_step_time)[:,self.id_proxy,:]]
        return [time, result]