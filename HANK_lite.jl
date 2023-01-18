#HANK model (Kaplan et al. 2018). (Limitaion: here is simple diffusion process verison).

cd(@__DIR__)
using Roots
using SparseArrays
using LinearAlgebra.BLAS
using Plots
Plots.gr()
using LaTeXStrings
using Distributions
using Parameters
using Interpolations


function two_asset_kinked_cost(d, a, χ₀, χ₁)
    return χ₀ .* abs.(d) .+ 0.5 * χ₁ .* d .^ 2 .* (max.(a, 10^(-5))) .^ (-1)
end

function two_asset_kinked_FOC(pa, pb, a, χ₀, χ₁)
    return min.(pa ./ pb .- 1 .+ χ₀, 0) .* a / χ₁ .+ max.(pa ./ pb .- 1 .- χ₀, 0) .* a / χ₁
end


@with_kw mutable struct Household
    τ::Real = 0.3 #Proportional labor tax
    T::Real = 0.0 #Lump sum transfer (rel GDP)
    ω::Real = 0.33#Share of proffit paid as liquid dividends
    Π::Real = 0 #profit

    #PARAMETERS
    γ::Real = 2 #coeffcient of relative risk aversion
    ϕ::Real = 0.02 #the Frisch elasticity of labor supply. 

    ra::Real = 0.05    #interest rate on asset
    rb_pos::Real = 0.03
    rb_neg::Real = 0.12 # r + κ
    ρ::Real = 0.051 #discount rate

    #adjustment cost function parameters:
    χ₀::Real = 0.045
    χ₁::Real = 0.956
    Χ₂::Real = 1.402


    #Income process (two-state Poisson process):
    w::Real = 4 #wage

    # ############## DIFFSION PROCESS #######################
    # # %ORNSTEIN-UHLENBECK IN LEVELS
    sig2::Real = 0.05
    Corr::Real = 0.9
    Nz::Int = 2
    zmin::Real = 0.8
    zmax::Real = 1.3
    dz = (zmax - zmin) / (Nz - 1)


    #grids
    I = 100
    bmin = -2
    bmax = 40
    J = 50
    amin = 0
    amax = 70

    b = nothing
    db = nothing
    a = nothing
    da = nothing
    g = nothing
    ga = nothing
    gb = nothing
    d = nothing
    m = nothing
    s = nothing
    v = nothing
    c = nothing
    l = nothing
    aaa = nothing
    bbb = nothing

    #total for asset, bond and labor:
    at = nothing
    bt = nothing
    lt = nothing
end

function Household_HJB_K(h::Household; crit=10^-6, Delta=100, maxit=500, lr=0.1)

  
    γ = h.γ
    ϕ = h.ϕ
    ra = h.ra
    rb_pos = h.rb_pos
    rb_neg = h.rb_neg

    ρ = h.ρ #discount rate
    τ = h.τ
    #adjustment cost function parameters:
    χ₀ = h.χ₀ #0.03
    χ₁ = h.χ₁ #2
    # Χ₂ = h.Χ₂

    if ra - 1 / χ₁ > 0
        println("Warning: ra - 1/χ₁ > 0")
    end


    ############## DIFFSION PROCESS #######################
    # %ORNSTEIN-UHLENBECK IN LEVELS
    sig2 = h.sig2
    Corr = h.Corr
    the = -log(Corr)
    logzmean = 0

    Nz = h.Nz
    zmin = h.zmin #0.8
    zmax = h.zmax #1.3

    z = range(zmin, zmax, Nz)
    # z=z/zmean; #### !!!!! chech this line 
    dz = (zmax - zmin) / (Nz - 1)
    h.dz = dz
    dz2 = dz^2

    mu = (the * (logzmean .- log.(z)) .+ sig2 / 2) .* z #DRIFT (FROM ITO'S LEMMA)
    s2 = sig2 .* z .^ 2 #VARIANCE (FROM ITO'S LEMMA)

    #grids
    I = h.I
    bmin = h.bmin #-2
    bmax = h.bmax #40
    b = range(bmin, bmax, I)
    db = (bmax - bmin) / (I - 1)

    h.b = b
    h.db = db

    J = h.J
    # J=3
    amin = h.amin #0
    amax = h.amax #70

    a = range(amin, amax, J)
    da = (amax - amin) / (J - 1)
    h.a = a
    h.da = da


    ###Income process (two-state Poisson process):
    w = (1 - h.τ) * h.w #take into accout tax
    T = h.T
    #Distribution of Profits
    π = zeros(I, J, Nz)
    for nz in 1:Nz
        π[:, :, nz] .= z[nz] / mean(z) * (1 - h.ω) * h.Π
    end

    # bb = b*ones(1,J);
    # aa = ones(I,1)*a';
    zz = ones(J, 1) * z'
    bbb = zeros(I, J, Nz)
    aaa = zeros(I, J, Nz)
    zzz = zeros(I, J, Nz)
    bbb[:, :, :] .= b * ones(1, J)
    aaa[:, :, :] .= ones(I, 1) * a'
    for nz in 1:Nz
        zzz[:, :, nz] .= z[nz]
    end

    # %CONSTRUCT MATRIX Bswitch SUMMARIZING EVOLUTION OF z
    χ = -min.(mu, 0) / dz .+ s2 / (2 * dz2)
    yy = min.(mu, 0) / dz .- max.(mu, 0) / dz .- s2 / dz2
    zeta = max.(mu, 0) / dz .+ s2 / (2 * dz2)


    updiag = zeros(I * J * Nz - I * J)
    centdiag = zeros(I * J * Nz)
    lowdiag = zeros(I * J * Nz - I * J)


    for i = 1:I*J
        centdiag[i] = χ[1] + yy[1]
        for nz = 1:Nz-1
            centdiag[I*J*nz+i] = yy[nz+1]
            lowdiag[I*J*(nz-1)+i] = χ[nz+1]
            updiag[I*J*(nz-1)+i] = zeta[nz]
        end
        centdiag[(Nz-1)*I*J+i] = yy[Nz] + zeta[Nz]
    end


    #Add up the upper, center, and lower diagonal into a sparse matrix
    # note: spdiagm(position => arrya), where position is 0 on diagonal, and +/- from diagonal
    Bswitch = spdiagm(0 => centdiag) + spdiagm(I * J => updiag) + spdiagm(-I * J => lowdiag)


    #Preallocation
    VbF = zeros(I, J, Nz)
    VbB = zeros(I, J, Nz)
    VaF = zeros(I, J, Nz)
    VaB = zeros(I, J, Nz)
    c = zeros(I, J, Nz)
    updiag = zeros(I * J, Nz)
    lowdiag = zeros(I * J, Nz)
    centdiag = zeros(I * J, Nz)
    AAi = [spzeros(I * J, I * J) for i in 1:Nz]#AAi = cell(Nz,1);
    BBi = [spzeros(I * J, I * J) for i in 1:Nz]#BBi = cell(Nz,1);
    Id_B = zeros(I, J, Nz)
    Id_F = zeros(I, J, Nz)
    d_B = zeros(I, J, Nz)
    d_F = zeros(I, J, Nz)

 

    function lab_solve(a, b, z, w, ra, rp, rn, γ, ϕ, π)
        # x0 = (w*z)^(ϕ*(1-γ)/(1+γ*ϕ));
        l_eq(l) = l - (w * z * l .+ T + π + (b > 0) * rp * b + (b < 0) * rn * b)^(-γ * ϕ) * ((1 - τ) .* w * z)^ϕ
        return find_zero(l_eq, (w * z)^(ϕ * (1 - γ) / (1 + γ * ϕ)))
    end
 

    #INITIAL GUESS

    l0 = zeros(I, J, Nz)
    l = zeros(I, J, Nz)
    for i in 1:I
        for j in 1:J
            for nz in 1:Nz
                l0[i, j, nz] = lab_solve(a[j], b[i], z[nz], w, ra, rb_pos, rb_neg, γ, ϕ, π[i, j, nz])
            end
        end
    end


    lmin = l0[1, 1, :]
    lmax = l0[I, J, :]

    if isnothing(h.v)
        v0 = ((w .* zzz .* l0 .+ T + π + ra .* aaa + rb_neg .* bbb) .^ (1 - γ)) / (1 - γ) / ρ
        v = v0
    else
        v = h.v
    end


    #return at different points in state space
    #matrix of liquid returns
    Rb = rb_pos .* (bbb .> 0) + rb_neg .* (bbb .< 0) #different interest rate for borrowing and lending 
    raa = ra .* ones(1, J)


    #if ra>>rb, impose tax on ra*a at high a, otherwise some households
    #accumulate infinite illiquid wealth (not needed if ra is close to or less than rb)
    tau = 10
    raa = ra .* (1 .- (1.33 .* amax ./ a) .^ (1 - tau)) #plot(a,raa.*a)
    #matrix of illiquid returns
    Ra = zeros(I, J, Nz)
    Ra[:, :, :] .= ones(I, 1) * raa'

    b_dist = zeros(I, maxit)
    a_dist = zeros(J, maxit)
    ab_dist = zeros(I, J, maxit)
    dist = zeros(maxit)


    # n=1
    for n in 1:maxit
        V = v
        #DERIVATIVES W.R.T. b
        # forward difference
        VbF[1:I-1, :, :] = (V[2:I, :, :] .- V[1:I-1, :, :]) / db
        # VbF[1:I-3, :, :] = (V[3:I-1, :, :] + V[4:I, :, :] .- V[1:I-3, :, :] - V[2:I-2, :, :]) / (2*db);
        # VbF[I-2:I-1, :, :] = (V[I-1:I, :, :] .- V[I-2:I-1, :, :]) / db;
        VbF[I, :, :] = (w * zzz[I, :, :] .* l0[I, :, :] .+ T .+ Rb[I, :, :] .* bmax) .^ (-γ) #state constraint boundary condition
        # backward difference
        VbB[2:I, :, :] = (V[2:I, :, :] - V[1:I-1, :, :]) / db
        # VbF[4:I, :, :] = (V[3:I-1, :, :] + V[4:I, :, :] .- V[1:I-3, :, :] - V[2:I-2, :, :]) / (2*db);
        # VbF[2:3, :, :] = (V[2:3, :, :] .- V[1:2, :, :]) / db;
        VbB[1, :, :] = (w * zzz[1, :, :] .* l0[1, :, :] .+ T + π[1, :, :] + Rb[1, :, :] .* bmin) .^ (-γ) #state constraint boundary condition


        #DERIVATIVES W.R.T. a
        # forward difference
        VaF[:, 1:J-1, :] = (V[:, 2:J, :] - V[:, 1:J-1, :]) / da
        # backward difference
        VaB[:, 2:J, :] = (V[:, 2:J, :] - V[:, 1:J-1, :]) / da

        ###useful quantities
        #optimal CONSUMPTION (FOC)
        c_B = max.(VbB, 10^(-32)) .^ (-1 / γ)# FOC u(c)' = V'
        l_B = (max.(VbB, 10^(-32)) .* (1 - τ) .* w .* zzz) .^ ϕ
        c_F = max.(VbF, 10^(-32)) .^ (-1 / γ)
        l_F = (max.(VbF, 10^(-32)) .* (1 - τ) .* w .* zzz) .^ ϕ

        # c_B = VbB.^ (-1 / γ) ;# FOC u(c)' = V'
        # l_B = (VbB.*w.*zzz).^ϕ;
        # c_F = VbF .^ (-1 / γ);
        # l_F = (VbF.*w.*zzz).^ϕ;

        ## optimal DEPOSITS eq(3) (FOC)
        dBB = two_asset_kinked_FOC(VaB, VbB, aaa, χ₀, χ₁)
        dFB = two_asset_kinked_FOC(VaB, VbF, aaa, χ₀, χ₁)
        #VaF(:,J,:) = VbB(:,J,:).*(1-ra.* χ₁ - χ₁*w*zzz(:,J,:)./a(:,J,:));
        dBF = two_asset_kinked_FOC(VaF, VbB, aaa, χ₀, χ₁)
        #VaF(:,J,:) = VbF(:,J,:).*(1-ra.*χ₁ - χ₁*w*zzz(:,J,:)./a(:,J,:));
        dFF = two_asset_kinked_FOC(VaF, VbF, aaa, χ₀, χ₁)


        #UPWIND SCHEME upwind makes a choice of forward or backward differences based on
        # the sign of the drift
        d_B = (dBF .> 0) .* dBF + (dBB .< 0) .* dBB
        #state constraints at amin and amax
        d_B[:, 1, :] = (dBF[:, 1, :] .> 10^(-32)) .* dBF[:, 1, :] #make sure d>=0 at amax, don't use VaB(:,1,:)
        d_B[:, J, :] = (dBB[:, J, :] .< -10^(-32)) .* dBB[:, J, :] #make sure d<=0 at amax, don't use VaF(:,J,:)
        d_B[1, 1, :] = max.(d_B[1, 1, :], 0)
        #split drift of b and upwind separately
        sc_B = w * zzz .* l_B .+ T + π + Rb .* bbb - c_B
        sd_B = (-d_B .- two_asset_kinked_cost(d_B, aaa, χ₀, χ₁))


        d_F = (dFF .> 0) .* dFF + (dFB .< 0) .* dFB
        #state constraints at amin and amax
        d_F[:, 1, :] = (dFF[:, 1, :] .> 10^(-32)) .* dFF[:, 1, :]#make sure d>=0 at amin, don't use VaB(:,1,:)
        d_F[:, J, :] = (dFB[:, J, :] .< -10^(-32)) .* dFB[:, J, :]#make sure d<=0 at amax, don't use VaF(:,J,:)

        #split drift of b and upwind separately
        sc_F = w * zzz .* l_F .+ T + π + Rb .* bbb - c_F
        sd_F = (-d_F - two_asset_kinked_cost(d_F, aaa, χ₀, χ₁))
        sd_F[I, :, :] = min.(sd_F[I, :, :], 0)


        Ic_B = (sc_B .< -10^(-32))
        Ic_F = (sc_F .> 10^(-32)) .* (1 .- Ic_B)
        Ic_0 = 1 .- Ic_F .- Ic_B

        Id_F = (sd_F .> 10^(-32))
        Id_B = (sd_B .< -10^(-32)) .* (1 .- Id_F)
        Id_B[1, :, :] .= 0
        Id_F[I, :, :] .= 0
        Id_B[I, :, :] .= 1 #don't use VbF at bmax so as not to pick up articial state constraint
        Id_0 = 1 .- Id_F .- Id_B

        c_0 = w * zzz .* l0 .+ T + π + Rb .* bbb

        c = c_F .* Ic_F + c_B .* Ic_B + c_0 .* Ic_0
        l = l_F .* Ic_F + l_B .* Ic_B + l0 .* Ic_0
        u = c .^ (1 - γ) / (1 - γ) - l .^ (1 + 1 / ϕ) / (1 + 1 / ϕ)


        #CONSTRUCT MATRIX BB SUMMARING EVOLUTION OF b
        X = -Ic_B .* sc_B / db - Id_B .* sd_B / db
        Y = (Ic_B .* sc_B - Ic_F .* sc_F) / db + (Id_B .* sd_B - Id_F .* sd_F) / db
        Z = Ic_F .* sc_F / db + Id_F .* sd_F / db

        # for i = 1:Nz
        #     centdiag(:,i) = reshape(Y(:,:,i),I*J,1);
        # end
        centdiag = reshape(Y, I * J, Nz)

        lowdiag[1:I-1, :] = X[2:I, 1, :]
        updiag[2:I, :] = Z[1:I-1, 1, :]

        for j = 2:J
            lowdiag[1:j*I, :] = [lowdiag[1:(j-1)*I, :]; X[2:I, j, :]; zeros(1, Nz)]
            updiag[1:j*I, :] = [updiag[1:(j-1)*I, :]; zeros(1, Nz); Z[1:I-1, j, :]]
        end

        for nz = 1:Nz
            BBi[nz] = spdiagm(0 => centdiag[:, nz]) .+ spdiagm(-1 => lowdiag[1:end-1, nz]) .+ spdiagm(1 => updiag[2:end, nz])
        end

        # BB = [BBi[1] spzeros(I * J, I * J); spzeros(I * J, I * J) BBi[2]]
        BB = [BBi[1] spzeros(I * J, I * J * (Nz - 1))]
        for nz = 2:Nz
            BB = [BB; spzeros(I * J, I * J * (nz - 1)) BBi[nz] spzeros(I * J, I * J * (Nz - nz))]
        end

        #CONSTRUCT MATRIX AA SUMMARIZING EVOLUTION OF a
        dB = Id_B .* dBB + Id_F .* dFB
        dF = Id_B .* dBF + Id_F .* dFF
        MB = min.(dB, 0)
        MF = max.(dF, 0) .+ w * zzz .* l .+ T + π + Ra .* aaa
        MB[:, J, :] = w * zzz[:, J, :] .* l[:, J, :] .+ T + π[:, J, :] + dB[:, J, :] + Ra[:, J, :] .* amax #this is hopefully negative
        MF[:, J, :] .= 0
        χ = -MB / da
        yy = (MB - MF) / da
        zeta = MF / da



        #MATRIX AAi
        for nz = 1:Nz
            #This will be the upperdiagonal of the matrix AAi
            # AAupdiag=zeros(I,1); #This is necessary because of the peculiar way spdiags is defined.
            AAupdiag = zeros(I * (J - 1))
            for j = 1:J-1
                # AAupdiag=[AAupdiag;zeta[:,j,nz]];
                AAupdiag[(j-1)*I+1:j*I] = zeta[:, j, nz]
            end

            #This will be the center diagonal of the matrix AAi
            AAcentdiag = yy[:, 1, nz]
            for j = 2:J-1
                AAcentdiag = [AAcentdiag; yy[:, j, nz]]
            end
            AAcentdiag = [AAcentdiag; yy[:, J, nz]]

            #This will be the lower diagonal of the matrix AAi
            AAlowdiag = χ[:, 2, nz]
            for j = 3:J
                AAlowdiag = [AAlowdiag; χ[:, j, nz]]
            end

            #Add up the upper, center, and lower diagonal into a sparse matrix
            AAi[nz] = spdiagm(0 => AAcentdiag) + spdiagm(-I => AAlowdiag) + spdiagm(I => AAupdiag)

        end

        # AA = [AAi[1] spzeros(I * J, I * J); spzeros(I * J, I * J) AAi[2]]

        AA = [AAi[1] spzeros(I * J, I * J * (Nz - 1))]
        for nz = 2:Nz
            AA = [AA; spzeros(I * J, I * J * (nz - 1)) AAi[nz] spzeros(I * J, I * J * (Nz - nz))]
        end

        A = AA + BB + Bswitch

        B = (1 / Delta + ρ) * spdiagm(0 => ones(I * J * Nz)) - A

        u_stacked = reshape(u, I * J * Nz, 1)
        V_stacked = reshape(V, I * J * Nz, 1)

        # vec = u_stacked + V_stacked / Delta
        # V_stacked = B \ vec #SOLVE SYSTEM OF EQUATIONS

        BLAS.axpy!(1/Delta,V_stacked,u_stacked)
        V_stacked = B\u_stacked; #SOLVE SYSTEM OF EQUATIONS

        V = reshape(V_stacked, I, J, Nz)


        Vchange = V - v
        v = lr * V + (1 - lr) * v
        # v=V

        # b_dist[:,n] = maximum(abs.(Vchange[:,:,2]);dims=2);
        b_dist[:, n] = maximum(maximum(abs.(Vchange); dims=3), dims=2)
        a_dist[:, n] = maximum(maximum(abs.(Vchange); dims=3), dims=1)
        ab_dist[:, :, n] = maximum(abs.(Vchange); dims=3)


        dist[n] = maximum(abs.(Vchange))
        println("Household Value Function, Iteration ", n, "; max Vchange = ", dist[n])
        if dist[n] < crit
            println("Household Value Function Converged, Iteration = ",n)
            break
        end

    end



    d = Id_B .* d_B + Id_F .* d_F
    m = d + w * zzz .* l .+ T + π + Ra .* aaa
    s = w * zzz .* l .+ T + π + Rb .* bbb - d - two_asset_kinked_cost(d, aaa, χ₀, χ₁) - c
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % STATIONARY DISTRIBUTION %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % RECONSTRUCT TRANSITION MATRIX WITH SIMPLER UPWIND SCHEME

    X = -min.(s, 0) / db
    Y = min.(s, 0) / db - max.(s, 0) / db
    Z = max.(s, 0) / db

    # for i = 1:Nz
    #     centdiag(:,i) = reshape(Y(:,:,i),I*J,1);
    # end
    centdiag = reshape(Y, I * J, Nz)

    lowdiag[1:I-1, :] = X[2:I, 1, :]
    updiag[2:I, :] = Z[1:I-1, 1, :]

    for j = 2:J
        lowdiag[1:j*I, :] = [lowdiag[1:(j-1)*I, :]; X[2:I, j, :]; zeros(1, Nz)]
        updiag[1:j*I, :] = [updiag[1:(j-1)*I, :]; zeros(1, Nz); Z[1:I-1, j, :]]
    end

    for nz = 1:Nz
        BBi[nz] = spdiagm(0 => centdiag[:, nz]) .+ spdiagm(-1 => lowdiag[1:end-1, nz]) .+ spdiagm(1 => updiag[2:end, nz])
    end

    # BB = [BBi[1] spzeros(I * J, I * J); spzeros(I * J, I * J) BBi[2]]
    BB = [BBi[1] spzeros(I * J, I * J * (Nz - 1))]
    for nz = 2:Nz
        BB = [BB; spzeros(I * J, I * J * (nz - 1)) BBi[nz] spzeros(I * J, I * J * (Nz - nz))]
    end


    # %CONSTRUCT MATRIX AA SUMMARIZING EVOLUTION OF a
    χ = -min.(m, 0) / da
    yy = min.(m, 0) / da - max.(m, 0) / da
    zeta = max.(m, 0) / da

    #MATRIX AAi
    for nz = 1:Nz
        #This will be the upperdiagonal of the matrix AAi
        # AAupdiag=zeros(I,1); #This is necessary because of the peculiar way spdiags is defined.
        AAupdiag = zeros(I * (J - 1))
        for j = 1:J-1
            # AAupdiag=[AAupdiag;zeta[:,j,nz]];
            AAupdiag[(j-1)*I+1:j*I] = zeta[:, j, nz]
        end

        #This will be the center diagonal of the matrix AAi
        AAcentdiag = yy[:, 1, nz]
        for j = 2:J-1
            AAcentdiag = [AAcentdiag; yy[:, j, nz]]
        end
        AAcentdiag = [AAcentdiag; yy[:, J, nz]]

        #This will be the lower diagonal of the matrix AAi
        AAlowdiag = χ[:, 2, nz]
        for j = 3:J
            AAlowdiag = [AAlowdiag; χ[:, j, nz]]
        end

        #Add up the upper, center, and lower diagonal into a sparse matrix
        AAi[nz] = spdiagm(0 => AAcentdiag) + spdiagm(-I => AAlowdiag) + spdiagm(I => AAupdiag)

    end

    # AA = [AAi[1] spzeros(I * J, I * J); spzeros(I * J, I * J) AAi[2]]

    AA = [AAi[1] spzeros(I * J, I * J * (Nz - 1))]
    for nz = 2:Nz
        AA = [AA; spzeros(I * J, I * J * (nz - 1)) AAi[nz] spzeros(I * J, I * J * (Nz - nz))]
    end

    A = AA + BB + Bswitch

    M = I * J * Nz

    # AT = A'
    # # % Fix one value so matrix isn't singular:
    # vec = zeros(M, 1)
    # iFix = 1657
    # # iFix = 2300
    # # iFix=round(Int,rand()*size(vec)[1])
    # vec[iFix] = 0.01
    # AT[iFix, :] = [zeros(1, iFix - 1) 1 zeros(1, M - iFix)]
    # g_stacked = AT \ vec


    AT = A'
    vec = ones(M) .* 10^-16
    g_stacked = AT \ vec
    g_sum = g_stacked' * ones(M, 1) * da * db * dz
    g_stacked = g_stacked ./ g_sum

    g = reshape(g_stacked, I, J, Nz)

    h.g = g
    h.d = Id_B .* d_B + Id_F .* d_F
    h.m = w * zzz .* l .+ T + π + Ra .* aaa
    h.s = w * zzz .* l .+ T + π + Rb .* bbb - d - two_asset_kinked_cost(d, aaa, χ₀, χ₁) - c

    h.v = v
    h.c = c
    h.l = l
    h.aaa = aaa
    h.bbb = bbb
    h.ga = sum(h.g[:, :, :], dims=1)[1, :, :] * h.db
    h.gb = sum(h.g[:, :, :], dims=2)[:, 1, :] * h.da

    h.bt = sum(h.gb' * h.b * h.db * h.dz)

    h.at = sum(h.ga' * h.a * h.da * h.dz)

    h.lt = sum(h.l .* h.g .* h.da .* h.db * h.dz)

end


function Output(K, N; α=0.33)
    return K^α * N^(1 - α)
end

function Ra(K, N; α=0.33, δ=0.07, m=0.9)
    return α * m * (N / K)^(1 - α) - δ
end

function Wage(K, N; α=0.33, m=0.9)
    return (1 - α) * m * (K / N)^α
end

function GovermentSpending(; w, N, rᵇ, Bᵍ, T, τ=0.25)
    return τ * w * N + rᵇ * Bᵍ - T
end

function Profit(Y; m=0.9)
    return (1 - m) * Y
end

function K_(; A, Y, ϵ=10)
    return A - Y / ϵ
end

function findEquilibrium(h; iter=30, crit=10^-5, lr=0.5,h_lr=0.5)

    #Solve Houshold problem wih initial parameters:
    Household_HJB_K(h; maxit=10, lr=h_lr, crit=10^-5)

    Y = Output(h.at, h.lt)
    K = K_(; A=h.at, Y=Y)
    N = h.lt
    h.ra = Ra(K, N)
    h.w = Wage(K, N)

    Π = (1 - m) * Y
    Bʰ = h.bt
    Bᵍ = Bᵍ_Y * Y
    # G = GovermentSpending(; w=wage(K, N), N=N, rᵇ=h.rb_pos, Bᵍ=Bᵍ, T=h.T)

    for i = 1:iter
        Household_HJB_K(h; maxit=10, lr=lr, crit=crit)
        K1 = K_(; A=h.at, Y=Y)
        Y1 = Output(K1, h.lt)
        Bᵍ1 = Bᵍ_Y * Y1
        Bʰ1 = h.bt
        N1 = h.lt
        ra1 = Ra(K1, N1)
        w1 = Wage(K1, N1)
        Π1 = (1 - m) * Y1
        # G1 = GovermentSpending(; w=w1, N=N1, rᵇ=h.rb_pos, Bᵍ=Bᵍ, T=h.T)

        Λ = abs(K1 - K) / K + abs(N1 - N) / N + abs((Bʰ1 - Bʰ) / Bʰ) + abs((Bᵍ1 - Bᵍ) / Bᵍ)

        h.ra = (1 - lr) * h.ra + lr * ra1
        h.w = (1 - lr) * h.w + lr * w1
        h.Π = Π1
        K = K1
        Y = Y1
        N = N1
        Bʰ = Bʰ1
        Bᵍ = Bᵍ1


        println("Iteration = ",i, "  Λ=",Λ);
        if Λ<crit
            println("Equilibrium have been found. Ineration  = ",i, "; Λ=",Λ);
            println("rᵃ = ", h.ra);
            println("wage = ",h.w);
            println("K = ",K);
            println("N = ",N);
            println("Bʰ = ",Bʰ );
            println("Bᴳ = ",Bᵍ );
            break;
        end
        println("____________________________________");

    end


end

### Mine Code###

#create initial household
h = Household();
# h.rb_neg=0.08;
h.rb_pos = 0.02;
h.rb_neg = h.rb_pos * 4;
h.ra = 0.028;
h.w = 1.7869;
h.Nz = 4;
h.sig2 = 0.05;
h.Corr = 0.9;
h.bmin = -5;
h.bmax = 20;
h.amin = 0;
h.amax = 70;
h.zmin = 0.7;
h.zmax = 1.5;
h.I = 200;
h.J = 100;
h.χ₀ = 0.03;
h.χ₁ = 0.9;

h.T = 0.06;
h.τ = 0.3;
h.Π = 0.2;
#some collibration parameters (accourding to the paper)
m = 0.9
Bᵍ_Y = -0.23 # Bᵍ to Y ration


findEquilibrium(h,iter=30)

