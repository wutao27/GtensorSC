module mymatrixfcn

export MyMatrixFcn

import Base.eltype, Base.size, Base.ndims, Base.issym

abstract AbstractMatrixFcn{T}

type MyMatrixFcn{T} <: AbstractMatrixFcn{T}
    m::Int
    n::Int
    mul::Function
end

eltype{T}(op::AbstractMatrixFcn{T}) = T

ndims(op::AbstractMatrixFcn) = 2

size(op::AbstractMatrixFcn, dim::Integer) = (dim == 1) ? op.m :
                                            (dim == 2) ? op.n : 1
size(op::AbstractMatrixFcn) = (op.m, op.n)

issym(op::AbstractMatrixFcn) = false

length(op::AbstractMatrixFcn) = op.m*op.n

*(op::AbstractMatrixFcn, b) = A_mul_B(op, b)

A_mul_B{R,S}(op::AbstractMatrixFcn{R}, b::AbstractVector{S}) = A_mul_B!(Array(promote_type(R,S), op.m), op, b)

A_mul_B!(output, op::AbstractMatrixFcn, b) = op.mul(output, b)

end