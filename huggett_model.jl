#Hugget model. HJB, KFE and  Equilibrium interest rate with the non-linear solver.
#Origin: https://benjaminmoll.com/codes/

cd(@__DIR__)
using SparseArrays
using LinearAlgebra.BLAS
using Plots
using LaTeXStrings
using BenchmarkTools
using Roots

mutable struct HuggetModel
    #PARAMETERS
    γ::Real     #CRRA utility with parameter s
    r::Real     #interest rate
    ρ::Real   #discount rate
    J::Int      # number of z points
    z::Vector{Real}
    la::Vector{Real}

    I::Int      #numer of points
    amin::Real  #borrowing constraint
    amax::Real
    a::StepRangeLen
    da::Real

    maxit::Int
    crit::Real
    Delta::Int

    c::Matrix{Real}
    v::Matrix{Real}
    g::Matrix{Real}
    A

    function HuggetModel(; γ=1.2, r=0.035, ρ=0.05, z, la, I=1000, amin, amax)
        J = size(z)[1]
        return new(γ, r, ρ, J, z, la,
            I, amin, amax, range(amin, amax, I), (amax - amin) / (I - 1),
            100, 10^-6, 1000,
            zeros(I, J), zeros(I, J), zeros(I * J, I * J), spzeros(I * J, I * J))
    end
end

function HJB_K(m::HuggetModel) #HJB and Kolmogorov Forward (Fokker-Planck) Equation
    aa = m.a * ones(1, m.J)
    zz = ones(m.I, m.J) .* m.z'


    dVf = zeros(m.I, m.J)
    dVb = zeros(m.I, m.J)

    #Add up the upper, center, and lower diagonal into a sparse matrix
    # note: spdiagm(position => arrya), where position is 0 on diagonal, and +/- from diagonal
    Aswitch = [spdiagm(0 => -m.la[1] * ones(m.I)) spdiagm(0 => m.la[1] * ones(m.I))
        spdiagm(0 => m.la[2] * ones(m.I)) spdiagm(0 => -m.la[2] * ones(m.I))]


    # INITIAL GUESS
    v0 = zeros(m.I, m.J)
    v0[:, 1] = (m.z[1] .+ m.r .* m.a) .^ (1 - m.γ) ./ (1 - m.γ) / m.ρ
    v0[:, 2] = (m.z[2] .+ m.r .* m.a) .^ (1 - m.γ) ./ (1 - m.γ) / m.ρ

    m.v = v0
    V_n = zeros(m.I, 2, m.maxit)
    dist = zeros(m.maxit)
    for n = 1:m.maxit
        V = m.v
        V_n[:, :, n] = V


        #  forward difference
        dVf[1:m.I-1, :] = (V[2:m.I, :] - V[1:m.I-1, :]) ./ m.da
        dVf[m.I, :] = (m.z .+ m.r .* m.amax) .^ (-m.γ) #will never be used, but impose state constraint a<=amax just in case

        # backward difference
        dVb[2:m.I, :] = (V[2:m.I, :] - V[1:m.I-1, :]) / m.da
        dVb[1, :] = (m.z .+ m.r .* m.amin) .^ (-m.γ) #state constraint boundary condition

        I_concave = dVb .> dVf #indicator whether value function is concave (problems arise if this is not the case)

        #consumption and savings with forward difference
        cf = dVf .^ (-1 / m.γ) # u'(c) = v'
        ssf = zz + m.r .* aa - cf

        #consumption and savings with backward difference
        cb = dVb .^ (-1 / m.γ) # u'(c) = v'
        ssb = zz + m.r .* aa - cb


        #consumption and derivative of value function at steady state
        c0 = zz + m.r .* aa
        dV0 = c0 .^ (-m.γ)


        # dV_upwind makes a choice of forward or backward differences based on
        # the sign of the drift    
        If = ssf .> 0 #positive drift --> forward difference
        Ib = ssb .< 0 #negative drift --> backward difference
        I0 = (1 .- If .- Ib) #at steady state
        #make sure backward difference is used at amax
        #Ib(I,:) = 1; If(I,:) = 0;
        #STATE CONSTRAINT at amin: USE BOUNDARY CONDITION UNLESS muf > 0:
        #already taken care of automatically

        dV_Upwind = dVf .* If + dVb .* Ib + dV0 .* I0 #important to include third term
        m.c = dV_Upwind .^ (-1 / m.γ)
        u = m.c .^ (1 - m.γ) / (1 - m.γ)

        #CONSTRUCT MATRIX  (HACT_Numerical_Appendix: eq(13-14))
        X = -min.(ssb, 0) / m.da
        Y = -max.(ssf, 0) / m.da + min.(ssb, 0) / m.da
        Z = max.(ssf, 0) / m.da

        #(eq 15)
        A1 = spdiagm(0 => Y[:, 1]) .+ spdiagm(-1 => X[2:m.I, 1]) .+ spdiagm(1 => Z[1:m.I-1, 1])
        A2 = spdiagm(0 => Y[:, 2]) .+ spdiagm(-1 => X[2:m.I, 2]) .+ spdiagm(1 => Z[1:m.I-1, 2])
        m.A = [A1 spzeros(m.I, m.I); spzeros(m.I, m.I) A2] + Aswitch


        B = (m.ρ + 1 / m.Delta) * spdiagm(0 => ones(2 * m.I)) - m.A


        u_stacked = reshape(u, m.I * m.J, 1)
        V_stacked = reshape(V, m.I * m.J, 1)

        # b = u_stacked + V_stacked/Delta;
        # V_stacked = B\b; #SOLVE SYSTEM OF EQUATIONS

        BLAS.axpy!(1 / m.Delta, V_stacked, u_stacked)
        V_stacked = B \ u_stacked #SOLVE SYSTEM OF EQUATIONS

        V = reshape(V_stacked, m.I, m.J)

        Vchange = V - m.v
        m.v = V

        dist[n] = maximum(abs.(Vchange))
        println("Household Value Function, Iteration ", n, "; max Vchange = ", dist[n])
        if dist[n] < m.crit
            println("Household Value Function Converged, Iteration = ",n);
            break
        end
    end


    ###############################
    # Kolmogorov Forward (Fokker-Planck) Equation
    AT = m.A'
    b = zeros(m.J * m.I)
    # b=rand(m.J*m.I)/10^12;
    #need to fix one value, otherwise matrix is singular
    i_fix = 1
    b[i_fix] = 0.1
    AT[i_fix, :] .= 0
    AT[i_fix, i_fix] = 1

    #Solve linear system
    gg = AT \ b
    # g_sum = gg'*ones(I*J,1)*da
    g_sum = sum(gg * m.da)
    gg = gg ./ g_sum
    m.g = reshape(gg, m.I, m.J)

    # g_sum = sum(0.5*(m.g[2:end,:]+m.g[1:end-1,:])*m.da)
    # m.g=m.g/g_sum
end

### Partial Equilibrium:
#create model
m = HuggetModel(r=0.0363, z=[0.106, 0.2], la=[1.5, 1.6], amin=-0.02, amax=3)

#solve model:
@time HJB_K(m);

### FIGURES###
#Wealth distribution
plot(m.a[1:120], m.g[1:120, :], label=[L"$g_1(a)$" L"g_2(a)"], size=(900, 600))
# plot(m.a[1:80],m.g[1:80,:],label=[L"$g_1(a)$" L"g_2(a)" ])
xlabel!(L"Wealth, $a$")
ylabel!(L"Densities, $g_j(a)$") 
savefig("./images/hugget_densites.png")

#Saving:
adot = ones(m.I, m.J) .* m.z' + m.r .* m.a * ones(1, m.J) - m.c;
plot(m.a[1:120], adot[1:120, :], label=[L"$s_1(a)$" L"s_2(a)"], size=(900, 600))
xlabel!(L"Wealth, $a$")
ylabel!(L"Saving, $s_j(a)$")
savefig("./images/hugget_saving.png")


### Equilibrium interest rate

function L(r)
    m.r = r
    HJB_K(m)
    s = sum(m.g' * m.a * m.da)
    global R = cat(R, [r s], dims=1)
    return s
end


R = zeros(0, 2)
m = HuggetModel(z=[0.1, 0.2], la=[1.5, 1.6], amin=-0.1, amax=3)
r = find_zero(L, (0.01, 0.045), xatol=1e-5, no_pts=20)

R = sortslices(R, dims=1)
plot(R[:, 1], R[:, 2], label="", color=1, size=(900, 600))
scatter!(R[:, 1], R[:, 2], label="", color=1)
scatter!([r], [0], label="", color=2)
yaxis!("Bond supply")
xaxis!("Interest rate")
# savefig("../../tex/images/hugget_eq_rate.png")
savefig("./images/hugget_eq_rate.png")

