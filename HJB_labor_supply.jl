#Consumption-Saving Problem with Endogenous Labor Supply
#origin: https://benjaminmoll.com/codes/

cd(@__DIR__)
using Roots
using Polynomials
using SparseArrays
using LinearAlgebra.BLAS
using Plots
using BenchmarkTools
using LaTeXStrings
using Parameters

function lab_solve(a, z, w, r, γ, ϕ)
    l_eq(l) = l - (w * z * l + r * a[1])^(-γ * ϕ) * (w * z)^ϕ
    return find_zero(l_eq, (w * z)^(ϕ * (1 - γ) / (1 + γ * ϕ)))
end

@with_kw mutable struct Household
    #PARAMETERS
    γ::Real = 2    #CRRA utility with parameter s
    r::Real = 0.03    #interest rate
    ρ::Real = 0.05 #discount rate
    J::Int = 2    # number of z points
    z::Vector{Real} = [0.1, 0.2]
    la::Vector{Real} = [1.5, 1]


    ϕ::Real = 0.5
    w::Real = 1.0

    I::Int = 500     #numer of points
    amin::Real = -0.15 #borrowing constraint
    amax::Real = 3
    a = range(amin, amax, I)
    da = (amax - amin) / (I - 1)

    c = nothing
    l = nothing
    v = nothing

end

function Household_HJB_K(h::Household; crit=10^-6, delta=100, maxit=50)

    γ = h.γ 
    r = h.r
    ρ = h.ρ 
    J = h.J       
    z1 = 0.1
    z2 = 0.2
    z = h.z
    la = h.la

    ϕ = h.ϕ

    w = h.w

    I = h.I
    amin = h.amin 
    amax = h.amax
    a = h.a
    da = h.da
    aa = [a a]
    zz = ones(I, 2) .* z'


    dVf = zeros(I, 2)
    dVb = zeros(I, 2)
    c = zeros(I, 2)
    l = zeros(I, 2)


    #Add up the upper, center, and lower diagonal into a sparse matrix
    # note: spdiagm(position => arrya), where position is 0 on diagonal, and +/- from diagonal
    Aswitch = [spdiagm(0 => -la[1] * ones(I)) spdiagm(0 => la[1] * ones(I))
        spdiagm(0 => la[2] * ones(I)) spdiagm(0 => -la[2] * ones(I))]

    l0 = [lab_solve.(a, z1, w, r, γ, ϕ) lab_solve.(a, z2, w, r, γ, ϕ)]

    # INITIAL GUESS

    v0 = zeros(I, 2)
    if isnothing(h.v)
        v0[:, 1] = (w * z[1] * l0[1, 1] .+ r .* a) .^ (1 - γ) ./ (1 - γ) / ρ
        v0[:, 2] = (w * z[2] * l0[1, 2] .+ r .* a) .^ (1 - γ) ./ (1 - γ) / ρ
        v = v0
    else
        v = h.v
    end


    lmin = l0[1, :]
    lmax = l0[I, :]

    V_n = zeros(I, 2, maxit)
    dist = zeros(maxit)
    A = zeros(I * J, I * J)
    for n = 1:maxit
        V = v
        V_n[:, :, n] = V

        #  forward difference
        dVf[1:I-1, :] = (V[2:I, :] - V[1:I-1, :]) ./ da
        dVf[I, :] = (w .* z .* lmax .+ r .* amax) .^ (-γ) #will never be used, but impose state constraint a<=amax just in case

        # backward difference
        dVb[2:I, :] = (V[2:I, :] - V[1:I-1, :]) / da
        dVb[1, :] = (w .* z .* lmin .+ r .* amin) .^ (-γ) #state constraint boundary condition

        #consumption and savings with forward difference
        cf = dVf .^ (-1 / γ)
        lf = (dVf .* w .* zz) .^ ϕ
        ssf = w .* zz .* lf + r .* aa - cf

        #consumption and savings with backward difference
        cb = dVb .^ (-1 / γ)
        lb = (dVb .* w .* zz) .^ ϕ
        ssb = w .* zz .* lb + r .* aa - cb


        #consumption and derivative of value function at steady state
        c0 = w .* zz .* l0 + r .* aa
        dV0 = c0 .^ (-γ)


        # dV_upwind makes a choice of forward or backward differences based on
        # the sign of the drift    
        If = ssf .> 0 #positive drift --> forward difference
        Ib = ssb .< 0 #negative drift --> backward difference
        I0 = (1 .- If .- Ib) #at steady state
        #make sure backward difference is used at amax
        #Ib(I,:) = 1; If(I,:) = 0;
        #STATE CONSTRAINT at amin: USE BOUNDARY CONDITION UNLESS muf > 0:
        #already taken care of automatically

        # dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0; #important to include third term
        # c = dV_Upwind.^(-1/γ);
        c = cf .* If + cb .* Ib + c0 .* I0
        l = lf .* If + lb .* Ib + l0 .* I0
        u = c .^ (1 - γ) / (1 - γ) - l .^ (1 + 1 / ϕ) / (1 + 1 / ϕ)

        #CONSTRUCT MATRIX  (HACT_Numerical_Appendix: eq(13-14))
        X = -Ib .* ssb / da
        Y = -If .* ssf / da + Ib .* ssb / da
        Z = If .* ssf / da

        #(eq 15)
        A1 = spdiagm(0 => Y[:, 1]) .+ spdiagm(-1 => X[2:I, 1]) .+ spdiagm(1 => Z[1:I-1, 1])
        A2 = spdiagm(0 => Y[:, 2]) .+ spdiagm(-1 => X[2:I, 2]) .+ spdiagm(1 => Z[1:I-1, 2])
        A = [A1 spzeros(I, I); spzeros(I, I) A2] + Aswitch


        B = (ρ + 1 / delta) * spdiagm(0 => ones(2 * I)) - A


        u_stacked = reshape(u, I * J, 1)
        V_stacked = reshape(V, I * J, 1)

        # b = u_stacked + V_stacked/delta;
        # V_stacked = B\b; #SOLVE SYSTEM OF EQUATIONS

        BLAS.axpy!(1 / delta, V_stacked, u_stacked)
        V_stacked = B \ u_stacked #SOLVE SYSTEM OF EQUATIONS

        V = reshape(V_stacked, I, J)

        Vchange = V - v
        v = V

        dist[n] = maximum(abs.(Vchange))
        println("Value Function, Iteration ", n, "max Vchange = ", dist[n])
        if dist[n] < crit
            println("Value Function Converged, Iteration = ", n)
            break
        end
    end

    ###############################
    # FOKKER-PLANCK EQUATION %
    AT = A'
    b = zeros(J * I)

    #need to fix one value, otherwise matrix is singular
    i_fix = 1
    b[i_fix] = 0.1
    AT[i_fix, :] .= 0
    AT[i_fix, i_fix] = 1

    #Solve linear system
    b[i_fix] = 0.001
    gg = AT \ b
    # g_sum = gg'*ones(I*J,1)*da
    g_sum = sum(gg * da)
    gg = gg ./ g_sum
    g = reshape(gg, I, J)
    # sum(g*da) #check the integral is equal one
    # plot(a,g)


    h.v = v
    h.c = c
    h.l = l

end


### Main code ####

h = Household()

Household_HJB_K(h; maxit=40);

###FIGURES#####

plot(h.a, h.c, label=[L"$c_1(a)$" L"c_2(a)"])
xlabel!(L"Wealth, $a$")
ylabel!(L"Consumption, $l_j(a)$")
title!("Consumption policy function")
savefig("./hugget_labor_consumption.png")


plot(h.a, h.l, label=[L"$l_1(a)$" L"l_2(a)"])
xlabel!(L"Wealth, $a$")
ylabel!(L"Label Supply, $l_j(a)$")
title!("Labor supply policy funtion")
savefig("./hugget_labor.png")

