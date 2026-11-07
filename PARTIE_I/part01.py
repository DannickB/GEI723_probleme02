from brian2 import *

N = 1000
taum = 10*ms
taupre = 10*ms
taupre2 = 30*ms
taupost = 10*ms
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
F = 15*Hz
gmax = .01
dApre = .015
dApre2 = -.008
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax
dApre2 *= gmax

eqs_neurons = '''
dv/dt = (ge * (Ee-v) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''

poisson_input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                      method='euler')
# Synapse model kept in variables so the STDP window plot below uses the exact same rule
syn_model = '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)
                dApre2/dt = -Apre2 / taupre2 : 1 (event-driven)
                '''
syn_on_pre = '''ge += w
                    Apre += dApre
                    Apre2 += dApre2
                    w = clip(w + Apost , 0, gmax)'''
syn_on_post = '''Apost += dApost
                     w = clip(w + Apre + Apre2, 0, gmax)'''

S = Synapses(poisson_input, neurons, syn_model,
             on_pre=syn_on_pre,
             on_post=syn_on_post,
             )
S.connect()
S.w = 'rand() * gmax'
mon = StateMonitor(S, 'w', record=[0, 1])
s_mon = SpikeMonitor(poisson_input)

run(100*second, report='text')

subplot(311)
plot(S.w / gmax, '.k')
ylabel('Weight / gmax')
xlabel('Synapse index')
subplot(312)
hist(S.w / gmax, 20)
xlabel('Weight / gmax')
subplot(313)
plot(mon.t/second, mon.w.T/gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()

# STDP window measured by simulation: one isolated pre/post spike pair per synapse,
# each pair with a different dt = t_post - t_pre, using syn_model/syn_on_pre/syn_on_post.
delta_t = arange(-100, 101, 1)*ms
n_pairs = len(delta_t)
t_ref = 150*ms
w_init = gmax / 2  # mid-range start so clip() only bites if a change exceeds gmax/2
stdp_pre = SpikeGeneratorGroup(n_pairs, arange(n_pairs), t_ref * ones(n_pairs))
stdp_post = NeuronGroup(n_pairs, '''ge : 1
                                    t_spike : second''',
                        threshold='timestep(t, dt) == timestep(t_spike, dt)', reset='')
stdp_post.t_spike = t_ref + delta_t
stdp_syn = Synapses(stdp_pre, stdp_post, syn_model,
                    on_pre=syn_on_pre, on_post=syn_on_post)
stdp_syn.connect(j='i')
stdp_syn.w = w_init
Network(stdp_pre, stdp_post, stdp_syn).run(2 * t_ref + max(abs(delta_t)))
delta_w = stdp_syn.w[:] - w_init

figure()
plot(delta_t/ms, delta_w)
axhline(0, color='gray', lw=0.5)
axvline(0, color='gray', lw=0.5)
xlabel(r'$\Delta t = t_{post} - t_{pre}$ (ms)')
ylabel(r'$\Delta w$')
title('STDP window')
print(f"dw(+20 ms) = {delta_w[delta_t == 20*ms][0]:.6g}")
print(f"dw(-20 ms) = {delta_w[delta_t == -20*ms][0]:.6g}")

show()