program GPU_DIRECT

   !ibmxl : mpif90 -O3 -qcuda GPU_DIRECT.F90 -o GPU_DIRECT.x

   use cudafor
   use ISO_C_BINDING
   implicit none
   include "mpif.h"
 
   integer :: nx,ny,nz,nv,mz,my,nsteps,tpn
   real, pinned, allocatable :: sndbuf(:,:,:,:,:),rcvbuf1(:,:,:,:,:),&
      rcvbuf2(:,:,:,:,:)
   real, device, allocatable :: d_sndbuf(:,:,:,:,:),d_rcvbuf2(:,:,:,:,:)
   integer :: i,x,y,z,zg,yg,z1,y1,ivar,istep
   integer :: numtasks,taskid,ierr
   integer :: isendbuf(6)
   real :: err(2),errmax(2)
   real*8 :: rtime1,rtime2,time(2),msgsize(2),avgtime,bw(2)
   real*8, allocatable :: time_step(:,:),time_all(:,:)

   call MPI_INIT (ierr)

   call MPI_COMM_SIZE (MPI_COMM_WORLD,numtasks,ierr)
   call MPI_COMM_RANK (MPI_COMM_WORLD,taskid,ierr)

!!!!!!!!! read input and communicate to all processes !!!!!!!!!!

   if(taskid.eq.0) then
      open (111,file='input')

      read (111,fmt='(1x)')
      read (111,*) nx,ny,nz,nv

      read (111,fmt='(1x)')
      read (111,*) tpn

      read (111,fmt='(1x)')
      read (111,*) nsteps
   end if
   isendbuf(1)=nx
   isendbuf(2)=ny
   isendbuf(3)=nz
   isendbuf(4)=nv
   isendbuf(5)=tpn
   isendbuf(6)=nsteps
   call MPI_BCAST (isendbuf,6,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
   nx=isendbuf(1)
   ny=isendbuf(2)
   nz=isendbuf(3)
   nv=isendbuf(4)
   tpn=isendbuf(5)
   nsteps=isendbuf(6)

   mz=nz/numtasks
   my=ny/numtasks
   if(taskid.eq.0)write(6,*) "  "
   if(taskid.eq.0)write(6,"('nx,ny,nz,nv,mz,my',6i6)") nx,ny,nz,nv,mz,my
   if(taskid.eq.0)write(6,"('nsteps,numtasks',2i6)") nsteps,numtasks
   if(taskid.eq.0)write(6,*) "  "

!!!!!!!! allocate arrays !!!!!!!!

   allocate(sndbuf(nx,my,mz,nv,numtasks))
   allocate(rcvbuf1(nx,my,mz,nv,numtasks))
   allocate(rcvbuf2(nx,my,mz,nv,numtasks))
   allocate(d_sndbuf(nx,my,mz,nv,numtasks))
   allocate(d_rcvbuf2(nx,my,mz,nv,numtasks))
   allocate(time_step(nsteps,2))
   allocate(time_all(2,numtasks))
   if(taskid.eq.0)write(6,"('Shape of array:',5i6)")shape(sndbuf)
   msgsize(1)=4.*nx*my*mz*nv*numtasks/1024./1024.
   msgsize(2)=4.*nx*my*mz*nv/1024./1024.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MPI-ALLTOALL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!! Initialise arrays !!!!!!!!

   time_step(:,:)=0.

   rcvbuf1(:,:,:,:,:)=0.
   rcvbuf2(:,:,:,:,:)=0.
   do yg=1,numtasks
      do ivar=1,nv
         do z=1,mz
            z1=taskid*mz+z
            do y=1,my
               y1=(yg-1)*my+y
               do x=1,nx
                  sndbuf(x,y,z,ivar,yg)=100.*x+10.*y1+1.*z1+0.1*ivar
               end do
            end do
         end do
      end do
   end do
   if(taskid.eq.0)write(6,*) "After Initialize"

!!!!!!!! Copy from device to host, Alltoall and copy back to device !!!!!!!

   do istep=1,nsteps

      call MPI_BARRIER(MPI_COMM_WORLD,ierr)
      rtime1=MPI_WTIME()

      ierr=cudaMemCpy(d_sndbuf,sndbuf,nx*my*mz*nv*numtasks)
      call MPI_ALLTOALL(sndbuf,nx*my*mz*nv,MPI_REAL,&
         rcvbuf1,nx*my*mz*nv,MPI_REAL,MPI_COMM_WORLD,ierr)
      ierr=cudaMemCpy(rcvbuf2,d_rcvbuf2,nx*my*mz*nv*numtasks)

      rtime2=MPI_WTIME()
      time_step(istep,1)=rtime2-rtime1

   end do
   if(taskid.eq.0)write(6,*) "After MPI-ALLTOALL"

!!!!!!!! GPU-Direct !!!!!!!

   ierr=cudaMemCpy(d_sndbuf,sndbuf,nx*my*mz*nv*numtasks)

   do istep=1,nsteps

      call MPI_BARRIER(MPI_COMM_WORLD,ierr)
      rtime1=MPI_WTIME()
      call MPI_ALLTOALL(d_sndbuf,nx*my*mz*nv,MPI_REAL,&
         d_rcvbuf2,nx*my*mz*nv,MPI_REAL,MPI_COMM_WORLD,ierr)
      rtime2=MPI_WTIME()
      time_step(istep,2)=rtime2-rtime1

   end do

   ierr=cudaMemCpy(rcvbuf2,d_rcvbuf2,nx*my*mz*nv*numtasks)

   if(taskid.eq.0)write(6,*) "After MPI-ALLTOALL (GPU-DIRECT)"

!!!!!!!!! Check the results !!!!!!!!
   
   do zg=1,numtasks
      do ivar=1,nv
         do z=1,mz
            z1=(zg-1)*mz+z
            do y=1,my
               y1=taskid*my+y
               do x=1,nx
                  rcvbuf1(x,y,z,ivar,zg)=rcvbuf1(x,y,z,ivar,zg)-&
                     (100.*x+10.*y1+1.*z1+0.1*ivar)
                  rcvbuf2(x,y,z,ivar,zg)=rcvbuf2(x,y,z,ivar,zg)-&
                     (100.*x+10.*y1+1.*z1+0.1*ivar)
               end do
            end do
         end do
      end do
   end do
   err(1)=maxval(abs(rcvbuf1))
   err(2)=maxval(abs(rcvbuf2))
   if(taskid.eq.0)write(6,*) "After Result check"

!!!!!!! Collect timings and errors from all tasks !!!!!!!!!

   time(1)=(sum(time_step(:,1))-time_step(1,1))/(nsteps-1)
   time(2)=(sum(time_step(:,2))-time_step(1,2))/(nsteps-1)
   call MPI_GATHER(time,2,MPI_REAL8,time_all,2,MPI_REAL8, &
      0,MPI_COMM_WORLD,ierr)
   call MPI_ALLREDUCE(err,errmax,2,MPI_REAL,MPI_MAX,MPI_COMM_WORLD,ierr)

!!!!!! Print the timings !!!!!!!!

   if (taskid.eq.0) then
      write(6,*) "  "
      write(6,"('Max Global Error              :',1p,e12.4)") errmax(1)
      write(6,"('Max Global Error (GPU-DIRECT) :',1p,e12.4)") errmax(2)
      write(6,*) "  "
      avgtime=sum(time_all(1,:))/numtasks
      bw(1)=2.*tpn*msgsize(1)/avgtime/1024.
      write(6,"('MPI-ALLTOALL (min,ave,max)     :',1p,3e12.4)") &
         minval(time_all(1,:)),avgtime,maxval(time_all(1,:))
      write(6,*) "  "
      avgtime=sum(time_all(2,:))/numtasks
      bw(2)=2.*tpn*msgsize(1)/avgtime/1024.
      write(6,"('GPU-DIRECT (min,ave,max)       :',1p,3e12.4)") &
         minval(time_all(2,:)),avgtime,maxval(time_all(2,:))
      write(6,*) "  "
      write(6,"('Total, P2P Message size (MB) :',2f12.2)")msgsize
      write(6,"('Eff. BW per node (bi-directional) :',2f8.2)") bw
      write(6,*) "  "
      write(6,*) "Taskid 0 :"
      write(6,*) "Step    MPI-ALLTOALL    GPU-DIRECT "
      write(6,*) "-------------------------------------"
      do i=1,nsteps
         write(6,"(i4,4x,1p,e12.4,2x,1p,e12.4)") i,&
            time_step(i,1),time_step(i,2)
      end do
      write(6,*) "  "
      write(6,*) "Taskid  MPI-ALLTOALL    GPU-DIRECT "
      write(6,*) "-------------------------------------"
      do i=1,numtasks
         write(6,"(i4,4x,1p,e12.4,2x,1p,e12.4)") i-1,&
            time_all(1,i),time_all(2,i)
      end do
   end if

   call MPI_FINALIZE (ierr)

end
